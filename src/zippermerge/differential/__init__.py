def hello() -> str:
    """
    Differential subpackage containing a discrete derivative function.
    """
    from .discrete import diff

    __all__ = [diff]
