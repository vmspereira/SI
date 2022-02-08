# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Dataset module"""
# ---------------------------------------------------------------------------
from .dataset import Dataset, summary
from .scale import StandardScaler
from .feature_selection import VarianceThreshold, SelectKBest
