"""Experimental discrete VESP gravity surrogate framework."""

from .models import DiscreteVESP, MultiShellDiscreteVESP
from .sources import SourceGeometry, SourceSet, fibonacci_sphere, make_shell_sources

__all__ = [
    "DiscreteVESP",
    "MultiShellDiscreteVESP",
    "SourceGeometry",
    "SourceSet",
    "fibonacci_sphere",
    "make_shell_sources",
]
