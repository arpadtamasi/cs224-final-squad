import dataclasses


def load_dataclass(data, data_class):
    try:
        fields = dataclasses.fields(data_class)
    except TypeError as x:
        return data

    fieldtypes = {f.name: f.type for f in fields}
    return data_class(**{f: load_dataclass(data[f], fieldtypes[f]) for f in data})


