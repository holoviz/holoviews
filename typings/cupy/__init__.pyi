# Partial stubs based on cupy 14.2.0
from typing import Any

import numpy as np

class ndarray:
    shape: tuple[int, ...]
    ndim: int
    dtype: np.dtype[Any]
    def __getattr__(self, name: str) -> Any: ...

def asnumpy(
    a: Any,
    stream: Any = None,
    order: str = "C",
    out: np.ndarray | None = None,
    *,
    blocking: bool = True,
) -> np.ndarray: ...
def asarray(
    a: Any,
    dtype: Any = None,
    order: str | None = None,
    *,
    copy: bool | None = None,
    blocking: bool = False,
) -> ndarray: ...
def histogram(
    x: ndarray, bins: Any = 10, range: Any = None, density: bool = False, weights: Any = None
) -> tuple[ndarray, ndarray]: ...
def percentile(
    a: ndarray,
    q: Any,
    axis: Any = None,
    out: ndarray | None = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
    *,
    interpolation: str | None = None,
) -> Any: ...
def isfinite(x: Any, /, *args: Any, **kwargs: Any) -> Any: ...
def __getattr__(name: str) -> Any: ...
