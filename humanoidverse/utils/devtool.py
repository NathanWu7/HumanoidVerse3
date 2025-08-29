

from typing import Callable, Any
from functools import wraps
def pdb_decorator(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)
    return wrapper


