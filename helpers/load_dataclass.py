import dataclasses
import typing

def load_dataclass(data, data_type):
    if not __is_data_class(data_type):
        return data

    try:
        if data is not None:
            real_type = __real_type(data_type)
            typed_data = {
                field.name: load_dataclass(data[field.name], field.type)
                for field in dataclasses.fields(real_type)
                if field.name in data
            }
            return real_type(**typed_data)
        elif __is_optional(data_type):
            return None
        else:
            return None


    except TypeError as t:
        raise Exception(f"Cannot load {data_type} from {data}:\n {t}") from t

def __is_data_class(data_type):
    try:
        dataclasses.fields(__real_type(data_type))
        return True
    except:
        return False

def __real_type(data_type):
    if __is_optional(data_type):
        return data_type.__args__[0]
    else:
        return data_type


def __is_optional(data_type):
    return (
            hasattr(data_type, "__args__")
            and len(data_type.__args__) == 2
            and data_type.__args__[-1] is type(None)
    )
