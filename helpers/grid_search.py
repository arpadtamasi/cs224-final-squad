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
        def generate(keys, acc):
            if keys:
                k, tail = keys[0], keys[1:]
                values = self.grid[k]
                for v in values:
                    acc = acc.copy()
                    acc[k] = v
                    yield from generate(tail, acc)
            else:
                yield self.__select_config(acc)

        yield from generate(list(self.grid.keys()), {})


    def __read_grid(self, config, path=''):
        for k in config.keys():
            key_path = f'{path}.{k}' if path else k
            v = config[k]
            if isinstance(v, dict):
                self.__read_grid(v, key_path)
            elif isinstance(v, list):
                self.grid[key_path] = v

    def __select_config(self, arguments):
        import copy
        import operator
        from functools import reduce
        config = copy.deepcopy(self.config)
        for path, value in [(p.split('.'), v) for p, v in arguments.items()]:
            parent_path = path[:-1]
            key = path[-1]
            parent = reduce(operator.getitem, parent_path, config)
            assert key in parent and isinstance(parent[key], list)
            parent[key] = value
        return (arguments, config)

    @staticmethod
    def experiment_path(experiment):
        return [f'{n}={experiment[n]}' for n in (sorted(list(experiment.keys())))]


