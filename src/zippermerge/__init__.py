from importlib.metadata import PackageNotFoundError, version
from .model import LinearRegression

try:
    __version__ = version("zippermerge")
except PackageNotFoundError:  # running from source without installation
    __version__ = "unknown"

__all__ = ["LinearRegression", "__version__"]