import importlib, sys
import typing
from .fc_cd import FC_EF
from .hrscd import HRSCD_str4
from .changeformer import ChangeFormer

__all__ = ["FC_EF", "HRSCD_str4"]

__mamba__ = {
    'CDMamba': 'cdresearch.models.CDMamba',
}

if typing.TYPE_CHECKING:
    from cdresearch.models.CDMamba import CDMamba

def __getattr__(name: str):
    if name in __mamba__:
        module_path = __mamba__[name]
        module = importlib.import_module(module_path)
        value = getattr(module, name) # Cache the value in the module's namespace for subsequent access
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__() -> typing.Iterable[str]:
    """
    Helps tools like IDEs and dir() discover the lazy attributes.
    """
    return sorted(list(globals()) + __all__)

