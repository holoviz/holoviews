try:
    # From Dask 2023.7,1 they now automatic convert strings
    # https://docs.dask.org/en/stable/changelog.html#v2023-7-1
    import dask
    dask.config.set({"dataframe.convert-string": False})
except Exception:
    pass
