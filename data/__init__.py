"""
data/__init__.py
Data module
"""

from .dataset import EnglishDialectsDataset, collate_fn

__all__ = [
    'EnglishDialectsDataset',
    'collate_fn'
]
