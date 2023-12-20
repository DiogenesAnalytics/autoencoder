"""Implements all minimal autoencoder architectures."""
__all__ = [
    "MinimalAutoencoder",
    "MinimalParams",
    "Minimal2DAutoencoder",
    "Min2DParams",
]

from .van2d import Min2DParams
from .van2d import Minimal2DAutoencoder
from .vanilla import MinimalAutoencoder
from .vanilla import MinimalParams
