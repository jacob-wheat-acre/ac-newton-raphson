"""dispatch.py — Generator thermodynamics and economic dispatch."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
from scipy.optimize import minimize, Bounds


class FuelType(Enum):
    NUCLEAR = auto()
    COAL    = auto()
    GAS_CC  = auto()
    GAS_CT  = auto()
    HYDRO   = auto()
    WIND    = auto()
    SOLAR   = auto()


FUEL_LABELS = {
    FuelType.NUCLEAR: "Nuclear",
    FuelType.COAL:    "Coal",
    FuelType.GAS_CC:  "Gas CC",
    FuelType.GAS_CT:  "Gas CT",
    FuelType.HYDRO:   "Hydro",
    FuelType.WIND:    "Wind",
    FuelType.SOLAR:   "Solar",
}

FUEL_COLORS = {
    FuelType.NUCLEAR: "#E65100",
    FuelType.COAL:    "#4E342E",
    FuelType.GAS_CC:  "#1565C0",
    FuelType.GAS_CT:  "#7B1FA2",
    FuelType.HYDRO:   "#00695C",
    FuelType.WIND:    "#2E7D32",
    FuelType.SOLAR:   "#F9A825",
}

# ton CO₂ / MBtu  (EPA eGRID defaults for combustion only)
CO2_FACTOR = {
    FuelType.NUCLEAR: 0.0,
    FuelType.COAL:    0.0948,
    FuelType.GAS_CC:  0.0531,
    FuelType.GAS_CT:  0.0531,
    FuelType.HYDRO:   0.0,
    FuelType.WIND:    0.0,
    FuelType.SOLAR:   0.0,
}

THERMAL_FUELS = {FuelType.NUCLEAR, FuelType.COAL, FuelType.GAS_CC, FuelType.GAS_CT}


@dataclass
class DispatchGen:
    name: str
    fuel: FuelType
    Pmin: float       # MW minimum output (must be online)
    Pmax: float       # MW rated/available capacity
    # Heat input curve: H(P) = hr_a + hr_b·P + hr_c·P²   [MBtu/hr]
    # For renewables and hydro, all three are zero.
    hr_a: float = 0.0   # MBtu/hr  no-load loss term
    hr_b: float = 0.0   # MBtu/MWh linear heat-rate coefficient
    hr_c: float = 0.0   # MBtu/(MW²·h) quadratic term (inefficiency at high load)
    fuel_cost: float = 0.0   # $/MBtu
    vom: float = 0.0         # $/MWh variable O&M

    # ── Thermodynamic methods ─────────────────────────────────────────────────
    def H(self, P: np.ndarray) -> np.ndarray:
        """Total heat input H(P) [MBtu/hr]."""
        return self.hr_a + self.hr_b * P + self.hr_c * P ** 2

    def HR(self, P: np.ndarray) -> np.ndarray:
        """Heat rate H(P)/P [MBtu/MWh] — lower is better."""
        safe_P = np.where(P > 1e-6, P, 1e-6)
        return np.where(P > 1e-6, self.H(P) / safe_P, self.hr_b)

    def eta(self, P: np.ndarray) -> np.ndarray:
        """Thermal efficiency (dimensionless): 3.412 MBtu/MWh ÷ HR."""
        hr = self.HR(P)
        return np.where(hr > 1e-9, 3.412 / hr, 0.0)

    # ── Cost methods ──────────────────────────────────────────────────────────
    def cost(self, P: np.ndarray) -> np.ndarray:
        """Total operating cost C(P) [$/hr]."""
        return self.fuel_cost * self.H(P) + self.vom * P

    def mc(self, P: np.ndarray) -> np.ndarray:
        """Marginal cost dC/dP [$/MWh]."""
        return self.fuel_cost * (self.hr_b + 2.0 * self.hr_c * P) + self.vom

    # ── Emissions ─────────────────────────────────────────────────────────────
    def co2_per_hr(self, P: np.ndarray) -> np.ndarray:
        """CO₂ emission rate [ton/hr]."""
        return CO2_FACTOR[self.fuel] * self.H(P)

    def co2_per_mwh(self, P: np.ndarray) -> np.ndarray:
        """CO₂ emission intensity [ton/MWh]."""
        return CO2_FACTOR[self.fuel] * self.HR(P)


def default_fleet() -> list[DispatchGen]:
    """
    Realistic mixed-technology generation fleet.

    Nuclear   — baseload, very low fuel cost, constant heat rate
    Coal      — mid-merit, quadratic heat-rate curve (Bergen & Vittal §5.4)
    CCGT      — mid-merit, efficient combined cycle
    Gas CT    — peaker, simple cycle, high heat rate
    Hydro     — zero fuel, low variable O&M (reservoir assumed available)
    Wind      — zero fuel, vom represents O&M / opportunity cost
    Solar     — zero fuel, vom represents O&M / opportunity cost
    """
    return [
        DispatchGen("Nuclear G1", FuelType.NUCLEAR,
                    Pmin=500, Pmax=1000,
                    hr_a=0,   hr_b=10.40, hr_c=0.0,
                    fuel_cost=0.50, vom=5.0),
        DispatchGen("Coal G2", FuelType.COAL,
                    Pmin=150, Pmax=500,
                    hr_a=510, hr_b=7.20, hr_c=0.00142,
                    fuel_cost=2.50, vom=4.0),
        DispatchGen("CCGT G3", FuelType.GAS_CC,
                    Pmin=50,  Pmax=400,
                    hr_a=150, hr_b=6.50, hr_c=0.0020,
                    fuel_cost=5.50, vom=3.0),
        DispatchGen("Gas CT G4", FuelType.GAS_CT,
                    Pmin=10,  Pmax=150,
                    hr_a=80,  hr_b=10.0, hr_c=0.008,
                    fuel_cost=5.50, vom=8.0),
        DispatchGen("Hydro H1", FuelType.HYDRO,
                    Pmin=0,   Pmax=100,
                    fuel_cost=0, vom=2.0),
        DispatchGen("Wind W1", FuelType.WIND,
                    Pmin=0,   Pmax=60,
                    fuel_cost=0, vom=15.0),
        DispatchGen("Solar S1", FuelType.SOLAR,
                    Pmin=0,   Pmax=40,
                    fuel_cost=0, vom=20.0),
    ]


def _merit_order_init(gens: list[DispatchGen], Pload_MW: float) -> np.ndarray:
    """Warm-start: fill cheapest generators first (merit order)."""
    Pmins = np.array([g.Pmin for g in gens])
    Pmaxs = np.array([g.Pmax for g in gens])
    P0 = Pmins.copy()
    remaining = Pload_MW - P0.sum()
    mc_at_min = [g.mc(np.array([g.Pmin]))[0] for g in gens]
    for i in sorted(range(len(gens)), key=lambda i: mc_at_min[i]):
        if remaining <= 1e-6:
            break
        add = min(Pmaxs[i] - P0[i], remaining)
        P0[i] += add
        remaining -= add
    return P0


def economic_dispatch(
    gens: list[DispatchGen],
    Pload_MW: float,
) -> tuple[np.ndarray, float, bool, str]:
    """
    Minimize Σ Cᵢ(Pᵢ)  s.t.  ΣPᵢ = Pload,  Pmin_i ≤ Pi ≤ Pmax_i.

    Returns (P_dispatch [MW array], total_cost [$/hr], converged, message).
    """
    n = len(gens)
    Pmins = np.array([g.Pmin for g in gens])
    Pmaxs = np.array([g.Pmax for g in gens])

    cap_min = float(Pmins.sum())
    cap_max = float(Pmaxs.sum())

    if Pload_MW < cap_min - 1e-3:
        return Pmins, float("nan"), False, (
            f"Load {Pload_MW:.1f} MW < committed minimum {cap_min:.1f} MW — "
            "reduce load or decommit a unit."
        )
    if Pload_MW > cap_max + 1e-3:
        return Pmaxs, float("nan"), False, (
            f"Load {Pload_MW:.1f} MW > available capacity {cap_max:.1f} MW — "
            "add generation or reduce load."
        )

    Pload_MW = float(np.clip(Pload_MW, cap_min, cap_max))
    P0 = _merit_order_init(gens, Pload_MW)

    def obj(P: np.ndarray) -> float:
        return float(sum(g.cost(np.array([P[i]]))[0] for i, g in enumerate(gens)))

    def jac(P: np.ndarray) -> np.ndarray:
        return np.array([g.mc(np.array([P[i]]))[0] for i, g in enumerate(gens)])

    res = minimize(
        obj, P0, method="SLSQP", jac=jac,
        bounds=Bounds(Pmins, Pmaxs),
        constraints={"type": "eq",
                     "fun": lambda P: P.sum() - Pload_MW,
                     "jac": lambda P: np.ones(n)},
        options={"ftol": 1e-10, "maxiter": 2000},
    )
    return res.x, float(res.fun), bool(res.success), str(res.message)
