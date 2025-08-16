
from typing import Dict, Callable

# Simple registries for model and hardware calculators
MODEL_CALCULATORS: Dict[str, Callable] = {}
HARDWARE_PROFILES: Dict[str, dict] = {}

def register_model(name: str):
    def deco(fn):
        MODEL_CALCULATORS[name] = fn
        return fn
    return deco

def register_hardware(name: str, profile: dict):
    HARDWARE_PROFILES[name] = profile
    return profile
