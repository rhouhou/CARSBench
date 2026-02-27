from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Literal
import numpy as np

@dataclass(frozen=True)
class Dist:
    kind: Literal["fixed", "uniform", "loguniform", "normal", "lognormal", "categorical"]
    params: Dict[str, Any]

    def sample(self, rng: np.random.Generator):
        k = self.kind
        p = self.params
        if k == "fixed":
            return p["value"]
        if k == "uniform":
            return rng.uniform(p["low"], p["high"])
        if k == "loguniform":
            low, high = float(p["low"]), float(p["high"])
            return float(np.exp(rng.uniform(np.log(low), np.log(high))))
        if k == "normal":
            return float(rng.normal(p["mean"], p["std"]))
        if k == "lognormal":
            return float(rng.lognormal(p["mean"], p["sigma"]))
        if k == "categorical":
            vals, probs = p["values"], p["probs"]
            idx = rng.choice(len(vals), p=probs)
            return vals[int(idx)]
        raise ValueError(f"Unknown Dist kind: {k}")

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "params": self.params}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Dist":
        return Dist(kind=d["kind"], params=dict(d["params"]))