"""Implements all minimal autoencoder architectures."""
__all__ = [
    "MinAE",
    "MinParams",
    "MinNDAE",
    "MinNDParams",
]

from .van2d import MinNDAE
from .van2d import MinNDParams
from .vanilla import MinAE
from .vanilla import MinParams
