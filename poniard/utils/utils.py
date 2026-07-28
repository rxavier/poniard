import inspect


def get_kwargs(exclude_self: bool = True, back: bool = False):
    """Returns the kwargs passed to the function/method that calls it."""
    frame = inspect.currentframe().f_back
    if back:
        frame = frame.f_back
    # co_varnames includes local vars in order; co_argcount limits to params
    params = frame.f_code.co_varnames[: frame.f_code.co_argcount]
    local_vars = frame.f_locals
    kwargs = {}
    for name in params:
        if exclude_self and name == "self":
            continue
        kwargs[name] = local_vars[name]
    return kwargs


def non_default_repr(obj):
    """Returns a class repr where only non-default params are included.

    The passed object needs to have the `_init_params` attribute, which
    is the result of running the `get_kwargs` function within initialization."""
    passed_kwargs = obj._init_params
    default_params = inspect.signature(obj.__class__).parameters
    default_params = {name: param.default for name, param in default_params.items()}
    non_default_params = {name: value for name, value in passed_kwargs.items()
                          if value != default_params[name]}
    params_string = ", ".join(
        [
            f"{k}={v}" if not isinstance(v, str) else f"{k}='{v}'"
            for k, v in non_default_params.items()
        ]
    )
    return f"""{obj.__class__.__name__}({params_string})"""
