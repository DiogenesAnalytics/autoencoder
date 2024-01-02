"""Implements all minimal autoencoder architectures."""
__all__ = [
    "MinAE",
    "MinParams",
    "Min2DAE",
    "Min2DParams",
]

from .van2d import Min2DAE
from .van2d import Min2DParams
from .vanilla import MinAE
from .vanilla import MinParams
