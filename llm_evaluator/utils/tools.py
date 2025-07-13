import os
from typing import Any


def load_api_key(cfgs: dict[str, Any]) -> dict[str, Any]:
    if "api_key" in cfgs:
        return cfgs
    elif "api_key_name" in cfgs:
        api_key_name = cfgs.pop("api_key_name")
        cfgs["api_key"] = os.environ.get(api_key_name)
        return cfgs
    else:
        return cfgs
