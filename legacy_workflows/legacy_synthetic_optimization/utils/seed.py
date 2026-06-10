from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """统一设置 Python 和 NumPy 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
