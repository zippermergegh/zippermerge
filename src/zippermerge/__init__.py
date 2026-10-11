from zippermerge._core import hello_from_bin
from importlib.metadata import PackageNotFoundError, version
from .model import LinearRegression

def hello() -> str:
    return hello_from_bin()


try:
    __version__ = version("hydroreg")
except PackageNotFoundError:  # running from source without installation
    __version__ = "unknown"

__all__ = ["LinearRegression", "__version__"]