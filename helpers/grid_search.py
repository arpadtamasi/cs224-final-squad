class GridSearch:
    def __init__(self, config):
        self.config = config
        self.grid = {}
        self.__read_grid(config)

    @staticmethod
    def load(path):
        from ujson import load as json_load
        with open(path, 'r') as pf: return GridSearch(json_load(pf))

    def random_experiments(self, max_experiments):
        import random
        all_experiments = list(self.experiments)
        random.shuffle(all_experiments)
        return all_experiments[:max_experiments]

    @property
    def experiments(self):
        from functools import reduce
        def extend(configs, experiments):
            experiment_key, alternatives = experiments
            return (
                {**config, **{experiment_key: alternative}}
                for config in configs
                for alternative in alternatives
            )

        experiment_settings = reduce(extend, self.grid.items(), [{}])
        return (self.experiment_config(settings) for settings in experiment_settings)

    def experiment_config(self, settings):
        import copy
        import operator
        from functools import reduce
        config = copy.deepcopy(self.config)
        for path, value in [(p.split('.'), v) for p, v in settings.items()]:
            parent_path = path[:-1]
            key = path[-1]
            parent = reduce(operator.getitem, parent_path, config)
            assert (key in parent)
            parent[key] = value
        return (settings, config)

    def __read_grid(self, config, path=''):
        for k in config.keys():
            key_path = f'{path}.{k}' if path else k
            v = config[k]
            if isinstance(v, dict):
                if v.keys() == {"hyperparam-search"}:
                    alternatives = v["hyperparam-search"]
                    assert isinstance(alternatives, list)
                    self.grid[key_path] = alternatives
                else:
                    self.__read_grid(v, key_path)

    @staticmethod
    def tags(experiment):
        def tags(path, value):
            if isinstance(value, dict):
                for k in value.keys():
                    yield from tags(f'{path}.{k}', value[k])
            else:
                yield (f'{path}={value}')

        return list(
            tag
            for key, value in (sorted(experiment.items()))
            for tag in tags(key, value)
        )
