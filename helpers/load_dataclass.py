import dataclasses


def load_dataclass(data, data_class):
    try:
        fields = {
            field.name: field.type
            for field in dataclasses.fields(data_class)
        }
    except TypeError as x:
        return data

    try:
        typed_data = {
            field_name: load_dataclass(data[field_name], field_type)
            for field_name, field_type in fields.items()
            if field_name in data
        }
        return data_class(**typed_data)
    except TypeError as t:
        raise Exception(f"Cannot load {data_class} from {data}:\n {t}") from t


