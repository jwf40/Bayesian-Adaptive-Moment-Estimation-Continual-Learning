def split_args(args: dict, keys: set= {'alg','ds','graduated'}):
    launch_kwargs = {}
    kwargs = {}
    for key in keys:
        launch_kwargs[key] = args[key]
        args.pop(key)
    kwargs = args
    return launch_kwargs, kwargs

def save():
    pass

