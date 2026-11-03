# Partial stubs based on cudf 26.08.1
from typing import Any

class DataFrame:
    def __init__(
        self,
        data: Any = None,
        index: Any = None,
        columns: Any = None,
        dtype: Any = None,
        copy: None = None,
        nan_as_null: Any = ...,
    ) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    def __getitem__(self, key: Any) -> Any: ...
    def __len__(self) -> int: ...

class Series:
    def __init__(
        self,
        data: Any = None,
        index: Any = None,
        dtype: Any = None,
        name: Any = None,
        copy: bool = False,
        nan_as_null: Any = ...,
    ) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    def __getitem__(self, key: Any) -> Any: ...
    def __len__(self) -> int: ...

def from_pandas(obj: Any, nan_as_null: Any = ...) -> Any: ...
def concat(
    objs: Any,
    axis: Any = 0,
    join: str = "outer",
    ignore_index: bool = False,
    keys: Any = None,
    levels: Any = None,
    names: Any = None,
    verify_integrity: bool = False,
    sort: bool | None = None,
) -> Any: ...
def __getattr__(name: str) -> Any: ...
