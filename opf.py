"""opf.py — DC Optimal Power Flow with Locational Marginal Pricing (LMP)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog


@dataclass
class OPFGen:
    bus: int
    Pmin: float    # MW
    Pmax: float    # MW
    c1: float      # $/MWh linear marginal cost
    name: str = ""


def _build_Bdc(nb: int, branches, bus_idx: dict) -> np.ndarray:
    """DC susceptance matrix (series reactance only, tap-adjusted)."""
    B = np.zeros((nb, nb))
    for br in branches:
        if abs(br.x) < 1e-9:
            continue
        i, j = bus_idx[br.fbus], bus_idx[br.tbus]
        tap = br.tap if br.tap > 1e-6 else 1.0
        b_ij = 1.0 / (br.x * tap)
        B[i, i] += b_ij
        B[j, j] += b_ij
        B[i, j] -= b_ij
        B[j, i] -= b_ij
    return B


def dc_opf(buses, branches, opf_gens: list[OPFGen],
           baseMVA: float = 100.0,
           enforce_limits: bool = True) -> dict:
    """
    Linear DC OPF via scipy linprog (HiGHS backend).

    Solves:
        min   Σ c1_k · Pg_k          [$/hr]
        s.t.  B·θ − Cg·Pg = −Pd      power balance at each bus [pu]
              Pmin_k ≤ Pg_k ≤ Pmax_k generator limits
              |P_ij| ≤ rate_j        line thermal limits (if enforce_limits)
              θ_slack = 0            angle reference

    LMP computation
    ───────────────
    b_eq = −Pd_pu  →  ∂b_eq_i / ∂Pd_MW_i = −1/baseMVA
    linprog marginals[i] = ∂cost/∂b_eq[i]
    ∴  LMP_i [$/MWh] = −marginals[i] / baseMVA

    Returns
    ───────
    dict with keys: Pg_mw, theta_rad, P_line_mw, LMP, total_cost,
                    converged, message
    """
    nb = len(buses)
    ng = len(opf_gens)
    bus_idx = {b.num: i for i, b in enumerate(buses)}
    slack_i = next(i for i, b in enumerate(buses) if b.type.name == "SLACK")

    Pd_pu = np.array([b.Pd / baseMVA for b in buses])

    # Generator-bus incidence [nb × ng]
    Cg = np.zeros((nb, ng))
    for k, og in enumerate(opf_gens):
        if og.bus in bus_idx:
            Cg[bus_idx[og.bus], k] = 1.0

    B = _build_Bdc(nb, branches, bus_idx)

    # Variables: x = [Pg_pu_0…ng-1, θ_0…nb-1]
    c_obj = np.zeros(ng + nb)
    for k, og in enumerate(opf_gens):
        c_obj[k] = og.c1 * baseMVA           # $/hr per pu injection

    lb = np.concatenate([[og.Pmin / baseMVA for og in opf_gens], [-np.pi] * nb])
    ub = np.concatenate([[og.Pmax / baseMVA for og in opf_gens], [ np.pi] * nb])
    lb[ng + slack_i] = ub[ng + slack_i] = 0.0   # fix reference angle

    # Equality: B·θ − Cg·Pg = −Pd  →  A_eq·x = b_eq
    A_eq = np.zeros((nb, ng + nb))
    A_eq[:, :ng] = -Cg
    A_eq[:, ng:] =  B
    b_eq = -Pd_pu

    # Inequality: |P_ij| ≤ rate  (two rows per rated branch)
    A_ub = b_ub = None
    if enforce_limits:
        rated = [(j, br) for j, br in enumerate(branches)
                 if br.rate > 0 and abs(br.x) > 1e-9]
        if rated:
            nr = len(rated)
            A_ub = np.zeros((2 * nr, ng + nb))
            b_ub = np.zeros(2 * nr)
            for row, (j, br) in enumerate(rated):
                fi = bus_idx[br.fbus]
                ti = bus_idx[br.tbus]
                tap = br.tap if br.tap > 1e-6 else 1.0
                b_ij = 1.0 / (br.x * tap)
                lim  = br.rate / baseMVA
                A_ub[row,      ng + fi] =  b_ij;  A_ub[row,      ng + ti] = -b_ij;  b_ub[row]      =  lim
                A_ub[nr + row, ng + fi] = -b_ij;  A_ub[nr + row, ng + ti] =  b_ij;  b_ub[nr + row] =  lim

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=list(zip(lb, ub)), method="highs")

    if res.success:
        Pg_mw  = res.x[:ng] * baseMVA
        theta  = res.x[ng:]
        LMP    = -res.eqlin.marginals / baseMVA   # $/MWh
        P_line = np.zeros(len(branches))
        for j, br in enumerate(branches):
            if abs(br.x) < 1e-9:
                continue
            fi = bus_idx[br.fbus]
            ti = bus_idx[br.tbus]
            tap = br.tap if br.tap > 1e-6 else 1.0
            P_line[j] = (theta[fi] - theta[ti]) / (br.x * tap) * baseMVA
        cost = float(res.fun)
    else:
        Pg_mw  = np.full(ng, float("nan"))
        theta  = np.zeros(nb)
        LMP    = np.zeros(nb)
        P_line = np.zeros(len(branches))
        cost   = float("nan")

    return dict(Pg_mw=Pg_mw, theta_rad=theta, P_line_mw=P_line,
                LMP=LMP, total_cost=cost, converged=res.success,
                message=res.message)


# ── Default OPF generator cost data ───────────────────────────────────────────
# Costs are representative $/MWh linear marginal costs (fuel + VOM).
# Varied enough across buses to produce interesting LMP maps under congestion.

def case_opf_gens(case_name: str) -> list[OPFGen] | None:
    """Return OPFGen list for a named case, or None for distribution feeders."""
    if any(k in case_name for k in ("33", "69")):
        return None   # radial distribution — no network-constrained redispatch

    if "39" in case_name:
        return [
            OPFGen( 30,   0,  400, c1=30.0, name="G30"),
            OPFGen( 31,   0, 1000, c1=25.0, name="G31 (slack)"),
            OPFGen( 32,   0,  800, c1=32.0, name="G32"),
            OPFGen( 33,   0,  700, c1=28.0, name="G33"),
            OPFGen( 34,   0,  650, c1=36.0, name="G34"),
            OPFGen( 35,   0,  750, c1=27.0, name="G35"),
            OPFGen( 36,   0,  700, c1=33.0, name="G36"),
            OPFGen( 37,   0,  700, c1=38.0, name="G37"),
            OPFGen( 38,   0, 1000, c1=22.0, name="G38"),
            OPFGen( 39,   0, 1100, c1=40.0, name="G39"),
        ]

    if "14" in case_name:
        return [
            OPFGen(1, 0, 332, c1=20.0, name="G1 (slack)"),
            OPFGen(2, 0, 140, c1=35.0, name="G2"),
            OPFGen(3, 0, 100, c1=30.0, name="G3"),
            OPFGen(6, 0, 100, c1=40.0, name="G6"),
            OPFGen(8, 0, 100, c1=45.0, name="G8"),
        ]

    if "9" in case_name:
        return [
            OPFGen(1, 10, 250, c1=20.0, name="G1 (slack)"),
            OPFGen(2, 10, 300, c1=40.0, name="G2"),
            OPFGen(3, 10, 270, c1=30.0, name="G3"),
        ]

    # 3-bus example
    return [
        OPFGen(1, 0, 200, c1=20.0, name="G1 (slack)"),
        OPFGen(2, 0, 250, c1=35.0, name="G2"),
        OPFGen(3, 0, 150, c1=25.0, name="G3"),
    ]
