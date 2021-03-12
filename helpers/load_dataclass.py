import dataclasses


def load_dataclass(data, data_class):
    try:
        fieldtypes = {f.name: f.type for f in dataclasses.fields(data_class)}
        return data_class(**{f: load_dataclass(data[f], fieldtypes[f]) for f in data})
    except:
        return data
