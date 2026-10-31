"""Default geometric and inertial parameters (paper Tables I and II)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Tuple

@dataclass
class RobotParams:
    rA: Tuple[float, float] = (0.0, 0.0)
    rB: Tuple[float, float] = (1.0, 0.0)
    rC: Tuple[float, float] = (1.0, 2.0)
    a: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    b: Tuple[float, float, float] = (0.3, 0.3, 0.3)
    c12: float = 0.15
    c23: float = 0.15
    c31: float = 0.15
    ma: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    mb: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    mG: float = 1.0
    Ia: Tuple[float, float, float] = (0.01, 0.01, 0.01)
    Ib: Tuple[float, float, float] = (0.01, 0.01, 0.01)
    IG: float = 0.01
    la: Tuple[float, float, float] = (0.05, 0.05, 0.05)
    lb: Tuple[float, float, float] = (0.15, 0.15, 0.15)


def validate_parameters(p: RobotParams) -> None:
    """Check whether a real, nondegenerate triangular platform is specified."""
    a, b, c = p.c12, p.c23, p.c31
    if min(a, b, c) <= 0 or not (a + b > c and a + c > b and b + c > a):
        raise ValueError("Platform sides must satisfy strict triangle inequalities.")
    for name in ("ma", "mb", "Ia", "Ib"):
        if len(getattr(p, name)) != 3 or min(getattr(p, name)) <= 0:
            raise ValueError(f"{name} must contain three strictly positive values.")
    if p.mG <= 0 or p.IG <= 0:
        raise ValueError("Platform mass and moment of inertia must be positive.")
    for name in ("a", "b", "la", "lb"):
        if len(getattr(p, name)) != 3 or min(getattr(p, name)) < 0:
            raise ValueError(f"{name} must contain three nonnegative values.")


def parameters_from_json(path: str | Path) -> RobotParams:
    """Override defaults using an explicit JSON file (never silent fallback)."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    allowed = set(RobotParams.__dataclass_fields__)
    unexpected = set(raw) - allowed
    if unexpected:
        raise ValueError(f"Unknown robot parameters: {sorted(unexpected)}")
    values = {name: tuple(v) if isinstance(v, list) else v for name, v in raw.items()}
    p = RobotParams(**values)
    validate_parameters(p)
    return p


def parameters_to_dict(p: RobotParams) -> dict:
    return asdict(p)
