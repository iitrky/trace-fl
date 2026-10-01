"""TRACE-FL: Temporal Reputation Aggregation with Clipping and Collusion-aware
Estimation for Byzantine-robust federated learning under non-IID data."""
from .aggregators import AGGREGATORS, get_aggregator
from .attacks import ATTACKS, apply_attack
from .simulation import run_fl

__version__ = "1.0.0"
__all__ = ["AGGREGATORS", "get_aggregator", "ATTACKS", "apply_attack", "run_fl"]
