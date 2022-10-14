import inspect
from typing import Dict, Any


def get_kwargs(exclude_self: bool = True):
    """Returns the kwargs passed to the function/method that calls it.

    Based on [this SO answer](https://stackoverflow.com/a/65927265/10840137)."""
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if exclude_self and key != "self":
            kwargs[key] = values[key]
    return kwargs


def get_non_default_params(obj, passed_kwargs: Dict[str, Any]):
    init_params = inspect.signature(obj).parameters
    defaults = {name: param.default for name, param in init_params.items()}
    return {name: value for name, value in passed_kwargs.items() if value != defaults[name]}
