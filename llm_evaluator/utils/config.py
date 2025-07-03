from typing import Any

import yaml  # type: ignore [import-untyped]

__all__ = [
    "load_config",
    "update_config_with_unparsed_args",
]


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load the configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict[str, Any]: Loaded configuration.
    """
    with open(config_path) as file:
        config: dict[str, Any] = yaml.safe_load(file)
    return config


def update_dict(
    total_dict: dict[str, Any], item_dict: dict[str, Any]
) -> dict[str, Any]:
    def update_dict(
        total_dict: dict[str, Any], item_dict: dict[str, Any]
    ) -> dict[str, Any]:
        for key, value in total_dict.items():
            if key in item_dict:
                total_dict[key] = item_dict[key]
            if isinstance(value, dict):
                update_dict(value, item_dict)
        return total_dict

    return update_dict(total_dict, item_dict)


def is_convertible_to_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def custom_cfgs_to_dict(key_list: str, value: Any) -> dict[str, Any]:
    """This function is used to convert the custom configurations to dict."""
    if value == "True":
        value = True
    elif value == "False":
        value = False
    elif value.isdigit():
        value = int(value)
    elif is_convertible_to_float(value):
        value = float(value)
    elif value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
        value = value.split(",")
        value = list(filter(None, value))
    elif "," in value:
        value = value.split(",")
        value = list(filter(None, value))
    else:
        value = str(value)
    keys_split = key_list.replace("-", "_").split(":")
    return_dict = {keys_split[-1]: value}

    for key in reversed(keys_split[:-1]):
        return_dict = {key.replace("-", "_"): return_dict}
    return return_dict


def update_config_with_unparsed_args(
    unparsed_args: list[str], cfgs: dict[str, Any]
) -> None:
    keys = [k[2:] for k in unparsed_args[::2]]
    values = list(unparsed_args[1::2])
    unparsed_args_dict = dict(zip(keys, values))

    for k, v in unparsed_args_dict.items():
        cfgs = update_dict(cfgs, custom_cfgs_to_dict(k, v))
