import inspect


def non_default_repr(obj):
    """Return a class repr showing only non-default constructor params.

    The passed object needs the `_init_params` attribute, a dict of the raw
    kwargs captured at the top of `__init__` (no frame introspection).
    """
    passed_kwargs = obj._init_params
    signature_params = inspect.signature(obj.__class__).parameters
    non_default_params = {}
    for name, value in passed_kwargs.items():
        if name not in signature_params:
            continue
        default = signature_params[name].default
        if default is inspect.Parameter.empty or value != default:
            non_default_params[name] = value
    params_string = ", ".join(
        f"{k}={v}" if not isinstance(v, str) else f"{k}='{v}'"
        for k, v in non_default_params.items()
    )
    return f"{obj.__class__.__name__}({params_string})"
