"""
data/__init__.py
데이터 모듈
"""

from .dataset import EnglishDialectsDataset, collate_fn

__all__ = [
    'EnglishDialectsDataset',
    'collate_fn'
]
