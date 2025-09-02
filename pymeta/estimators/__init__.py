"""Tau² estimators with registry pattern."""

from .tau2_dl import tau2_dersimonian_laird
from .tau2_pm import tau2_paule_mandel
from .tau2_reml import tau2_reml

__all__ = ['tau2_dersimonian_laird', 'tau2_paule_mandel', 'tau2_reml']