"""stability.py — Classical model transient stability (swing equation)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from acpf_nr import build_ybus


@dataclass
class StabGen:
    bus: int
    H: float      # inertia constant [s] on 100 MVA system base
    Xd: float     # d-axis transient reactance [pu] on system base
    name: str = ""


# ── Default classical-model generator data ─────────────────────────────────────

def default_stab_gens(case_name: str) -> list[StabGen] | None:
    """
    Return StabGen list for each standard case, or None for unsupported cases.

    9-bus:  Anderson & Fouad (1977) / MATPOWER values, 100 MVA base.
    14-bus: Approximate — no canonical stability dataset.
    39-bus: Approximate New England values; bus 31 is the external equivalent
            (very large inertia).
    """
    if any(k in case_name for k in ("33", "69")):
        return None

    if "39" in case_name:
        return [
            StabGen(30, H= 5.0, Xd=0.200, name="G30"),
            StabGen(31, H=30.0, Xd=0.006, name="G31 (ext. equiv.)"),
            StabGen(32, H= 4.0, Xd=0.060, name="G32"),
            StabGen(33, H= 3.5, Xd=0.070, name="G33"),
            StabGen(34, H= 3.2, Xd=0.090, name="G34"),
            StabGen(35, H= 3.8, Xd=0.080, name="G35"),
            StabGen(36, H= 4.0, Xd=0.090, name="G36"),
            StabGen(37, H= 3.0, Xd=0.100, name="G37"),
            StabGen(38, H= 4.5, Xd=0.120, name="G38"),
            StabGen(39, H= 7.0, Xd=0.040, name="G39"),
        ]

    if "14" in case_name:
        return [
            StabGen(1, H=10.0, Xd=0.10, name="G1"),
            StabGen(2, H= 3.0, Xd=0.22, name="G2"),
            StabGen(3, H= 2.0, Xd=0.25, name="G3 (sync)"),
            StabGen(6, H= 2.0, Xd=0.25, name="G6 (sync)"),
            StabGen(8, H= 2.0, Xd=0.25, name="G8 (sync)"),
        ]

    if "9" in case_name:
        # Anderson & Fouad (1977) / MATPOWER values on 100 MVA system base
        return [
            StabGen(1, H=23.64, Xd=0.0608, name="G1"),
            StabGen(2, H= 6.40, Xd=0.1198, name="G2"),
            StabGen(3, H= 3.01, Xd=0.1813, name="G3"),
        ]

    # 3-bus (2 generators)
    return [
        StabGen(1, H=10.0, Xd=0.12, name="G1"),
        StabGen(2, H= 4.0, Xd=0.20, name="G2"),
    ]


# ── Classical-model reduced admittance matrix ──────────────────────────────────

def _build_Yred(
    buses, branches, stab_gens: list[StabGen],
    Vm: np.ndarray, Va: np.ndarray,
    fault_bus: int | None = None,
    drop_branch_idx: int | None = None,
    baseMVA: float = 100.0,
) -> np.ndarray:
    """
    Build the generator-internal reduced admittance matrix (ng × ng).

    Algorithm
    ─────────
    1. Network Ybus (optionally with one branch removed for post-fault network).
    2. Convert all bus loads to constant admittances at pre-fault voltage.
    3. If fault_bus given: add large shunt at that bus (models V = 0 during fault).
    4. Augment Ybus with ng internal generator buses connected through j·X'd.
    5. Kron-reduce: eliminate all physical buses, retain generator internal buses.
    """
    nb  = len(buses)
    ng  = len(stab_gens)
    bus_idx = {b.num: i for i, b in enumerate(buses)}

    active = ([br for k, br in enumerate(branches) if k != drop_branch_idx]
               if drop_branch_idx is not None else list(branches))
    Y = build_ybus(nb, active, {b.num: b for b in buses}).astype(complex)

    # Constant-impedance loads
    for i, b in enumerate(buses):
        V2 = Vm[i] ** 2
        if V2 > 1e-12:
            Y[i, i] += (b.Pd - 1j * b.Qd) / (baseMVA * V2)

    # Short the faulted bus to ground
    if fault_bus is not None:
        fi = bus_idx[fault_bus]
        Y[fi, fi] += 1e6 + 0j

    # Augment: generator internal buses at indices nb … nb+ng-1
    Y_aug = np.zeros((nb + ng, nb + ng), dtype=complex)
    Y_aug[:nb, :nb] = Y
    for k, sg in enumerate(stab_gens):
        ti = bus_idx[sg.bus]
        gi = nb + k
        y_k = 1.0 / (1j * sg.Xd)
        Y_aug[ti, ti] += y_k
        Y_aug[gi, gi] += y_k
        Y_aug[ti, gi] -= y_k
        Y_aug[gi, ti] -= y_k

    # Kron reduction: Y_red = Y_gg − Y_gl · Y_ll⁻¹ · Y_lg
    Y_gg = Y_aug[nb:, nb:]
    Y_gl = Y_aug[nb:, :nb]
    Y_ll = Y_aug[:nb, :nb]
    Y_lg = Y_aug[:nb, nb:]
    try:
        X = np.linalg.solve(Y_ll, Y_lg)
    except np.linalg.LinAlgError:
        X = np.linalg.lstsq(Y_ll, Y_lg, rcond=None)[0]
    return Y_gg - Y_gl @ X


# ── Electrical power from classical model ──────────────────────────────────────

def _Pe(E_mag: np.ndarray, delta: np.ndarray, Yred: np.ndarray) -> np.ndarray:
    """
    Pe_i = E_i² G_ii + Σ_{j≠i} E_i E_j (G_ij cos Δδ + B_ij sin Δδ)
    where G + jB = Y_red.
    """
    ng = len(E_mag)
    Pe = E_mag ** 2 * Yred.diagonal().real
    for i in range(ng):
        for j in range(ng):
            if i != j:
                dij = delta[i] - delta[j]
                Pe[i] += (E_mag[i] * E_mag[j] *
                          (Yred[i, j].real * np.cos(dij) +
                           Yred[i, j].imag * np.sin(dij)))
    return Pe


# ── Swing equation RHS ─────────────────────────────────────────────────────────

def _swing(t: float, y: np.ndarray, ng: int, Yred: np.ndarray,
           E_mag: np.ndarray, Pm: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    State: y = [δ₁…δₙ, Δω₁…Δωₙ]  (δ in rad, Δω in rad/s)
    dδ_i/dt  = Δω_i
    dΔω_i/dt = (ω_s / 2H_i) (Pm_i − Pe_i)    ω_s = 2π·60
    """
    delta  = y[:ng]
    domega = y[ng:]
    Pe     = _Pe(E_mag, delta, Yred)
    return np.concatenate([domega,
                           (2 * np.pi * 60.0 / (2.0 * H)) * (Pm - Pe)])


# ── Instability check ──────────────────────────────────────────────────────────

def _is_unstable(delta_all: np.ndarray, threshold_deg: float = 120.0) -> bool:
    """True if any generator deviates > threshold from the unweighted COI."""
    coi = delta_all.mean(axis=0)                        # (n_t,)
    return bool(np.any(np.abs(delta_all - coi) > np.deg2rad(threshold_deg)))


# ── Main simulation ────────────────────────────────────────────────────────────

def run_stability(
    buses, branches, stab_gens: list[StabGen],
    Vm: np.ndarray, Va: np.ndarray,
    Pg_solved: np.ndarray, Qg_solved: np.ndarray,
    fault_bus: int,
    t_clear: float,
    drop_branch_idx: int | None = None,
    t_end: float = 5.0,
    baseMVA: float = 100.0,
    H_override: np.ndarray | None = None,
) -> dict:
    """
    Simulate a three-phase fault at fault_bus cleared at t_clear.

    Phases
    ──────
    0 → t_clear : during-fault (Y_red with fault bus shorted)
    t_clear → t_end: post-fault (Y_red with optional line removed)

    Returns
    ───────
    t           : 1-D time array [s]
    delta_deg   : (ng, len(t)) rotor angles [°]
    delta_rel   : (ng, len(t)) angles relative to G1 [°]
    delta0_deg  : initial rotor angles [°] (1-D, length ng)
    Pm_pu       : mechanical power [pu] (1-D, length ng)
    E_mag       : internal voltage magnitudes [pu]
    H           : inertia constants [s]
    gen_names   : list of strings
    stable      : bool
    t_clear     : echo
    fault_bus   : echo
    """
    ng      = len(stab_gens)
    bus_idx = {b.num: i for i, b in enumerate(buses)}
    H       = H_override if H_override is not None else np.array([sg.H for sg in stab_gens])

    # ── Pre-fault initial conditions ──────────────────────────────────────────
    E_complex = np.zeros(ng, dtype=complex)
    Pm        = np.zeros(ng)

    for k, sg in enumerate(stab_gens):
        i       = bus_idx[sg.bus]
        Va_rad  = np.deg2rad(Va[i])
        Vk      = Vm[i] * np.exp(1j * Va_rad)
        Sk_pu   = (Pg_solved[i] + 1j * Qg_solved[i]) / baseMVA
        Ik      = np.conj(Sk_pu / Vk)          # generator current injection [pu]
        E_complex[k] = Vk + 1j * sg.Xd * Ik   # internal voltage phasor
        Pm[k]        = Pg_solved[i] / baseMVA  # mech. input = pre-fault Pe

    E_mag  = np.abs(E_complex)
    delta0 = np.angle(E_complex)               # initial rotor angles [rad]
    y0     = np.concatenate([delta0, np.zeros(ng)])   # Δω₀ = 0

    # ── Reduced admittance matrices ────────────────────────────────────────────
    Yr_fault = _build_Yred(buses, branches, stab_gens, Vm, Va,
                            fault_bus=fault_bus, drop_branch_idx=None,
                            baseMVA=baseMVA)
    Yr_post  = _build_Yred(buses, branches, stab_gens, Vm, Va,
                            fault_bus=None, drop_branch_idx=drop_branch_idx,
                            baseMVA=baseMVA)

    n_pts   = 800
    t_ev1   = np.linspace(0,       t_clear, max(int(n_pts * t_clear / t_end), 30))
    t_ev2   = np.linspace(t_clear, t_end,   max(int(n_pts * (t_end - t_clear) / t_end), 60))

    def rhs(Yr):
        return lambda t, y: _swing(t, y, ng, Yr, E_mag, Pm, H)

    sol1 = solve_ivp(rhs(Yr_fault), [0.0, t_clear], y0, t_eval=t_ev1,
                     method="RK45", rtol=1e-7, atol=1e-9)
    sol2 = solve_ivp(rhs(Yr_post),  [t_clear, t_end], sol1.y[:, -1],
                     t_eval=t_ev2,  method="RK45", rtol=1e-7, atol=1e-9)

    t_all     = np.concatenate([sol1.t, sol2.t])
    delta_all = np.hstack([sol1.y[:ng, :], sol2.y[:ng, :]])  # (ng, n_t)

    delta_deg = np.rad2deg(delta_all)
    delta_rel = delta_deg - delta_deg[0:1, :]   # relative to G1

    stable = not _is_unstable(delta_all)

    return dict(
        t=t_all,
        delta_deg=delta_deg,
        delta_rel=delta_rel,
        delta0_deg=np.rad2deg(delta0),
        Pm_pu=Pm,
        E_mag=E_mag,
        H=H,
        gen_names=[sg.name or f"G{sg.bus}" for sg in stab_gens],
        stable=stable,
        t_clear=t_clear,
        fault_bus=fault_bus,
    )


# ── Critical clearing time (bisection) ────────────────────────────────────────

def find_cct(
    buses, branches, stab_gens: list[StabGen],
    Vm: np.ndarray, Va: np.ndarray,
    Pg_solved: np.ndarray, Qg_solved: np.ndarray,
    fault_bus: int,
    drop_branch_idx: int | None = None,
    t_lo: float = 0.02,
    t_hi: float = 2.0,
    tol: float = 0.01,
    t_end: float = 5.0,
    baseMVA: float = 100.0,
    H_override: np.ndarray | None = None,
) -> dict:
    """
    Bisect for the critical clearing time.

    Returns dict with 'cct' [s], 'stable_at_lo', 'stable_at_hi'.
    """
    def _stable(tc: float) -> bool:
        r = run_stability(buses, branches, stab_gens, Vm, Va,
                          Pg_solved, Qg_solved, fault_bus, tc,
                          drop_branch_idx=drop_branch_idx,
                          t_end=t_end, baseMVA=baseMVA,
                          H_override=H_override)
        return r["stable"]

    stable_lo = _stable(t_lo)
    stable_hi = _stable(t_hi)

    if stable_hi:
        return dict(cct=t_hi, stable_at_lo=stable_lo, stable_at_hi=True,
                    note=f"CCT > {t_hi:.2f} s — stable even at maximum clearing time")

    if not stable_lo:
        return dict(cct=t_lo, stable_at_lo=False, stable_at_hi=False,
                    note=f"Unstable even at {t_lo:.2f} s — check fault location or pre-fault state")

    while (t_hi - t_lo) > tol:
        tm = (t_lo + t_hi) / 2.0
        if _stable(tm):
            t_lo = tm
        else:
            t_hi = tm

    cct = (t_lo + t_hi) / 2.0
    return dict(cct=cct, stable_at_lo=stable_lo, stable_at_hi=stable_hi,
                note=f"CCT ≈ {cct:.3f} s  (±{tol/2:.3f} s)")


# ── DER penetration modelling ──────────────────────────────────────────────────

def effective_H(
    stab_gens: list[StabGen],
    der_penetration: float,
    der_type: str,
    H_virtual: float = 3.0,
) -> np.ndarray:
    """
    Return modified inertia constants [s] for a given DER scenario.

    der_penetration : 0–1, fraction of generation capacity displaced by DERs
    der_type        : "none" | "legacy" | "1547-2018"
    H_virtual       : virtual inertia constant synthesised by 1547-2018 inverters [s]
                      (grid-forming or fast-frequency-response capability)

    Legacy DERs displace synchronous inertia and contribute nothing.
    1547-2018 DERs displace synchronous inertia but synthesise H_virtual per unit
    of displaced capacity, distributed proportionally to original H.
    """
    H_orig  = np.array([sg.H for sg in stab_gens], dtype=float)
    H_total = H_orig.sum()
    pen     = float(np.clip(der_penetration, 0.0, 1.0))

    if der_type in ("none", "0"):
        return H_orig.copy()

    if der_type == "legacy":
        # Pure displacement — no virtual inertia
        return H_orig * (1.0 - pen)

    # IEEE 1547-2018: displace synchronous H, add back virtual inertia
    # Virtual contribution for each generator proportional to original share
    H_virt_contrib = H_virtual * pen * (H_orig / H_total)
    return H_orig * (1.0 - pen) + H_virt_contrib


def compute_rocof(
    stab_gens: list[StabGen],
    buses, branches,
    Vm: np.ndarray, Va: np.ndarray,
    Pg_solved: np.ndarray, Qg_solved: np.ndarray,
    fault_bus: int,
    H_eff: np.ndarray,
    baseMVA: float = 100.0,
) -> dict:
    """
    ROCOF immediately after fault inception [Hz/s] for each generator
    and inertia-weighted system ROCOF.
    """
    ng      = len(stab_gens)
    bus_idx = {b.num: i for i, b in enumerate(buses)}

    E_complex = np.zeros(ng, dtype=complex)
    Pm        = np.zeros(ng)
    for k, sg in enumerate(stab_gens):
        i  = bus_idx[sg.bus]
        Vk = Vm[i] * np.exp(1j * np.deg2rad(Va[i]))
        Sk = (Pg_solved[i] + 1j * Qg_solved[i]) / baseMVA
        Ik = np.conj(Sk / Vk)
        E_complex[k] = Vk + 1j * sg.Xd * Ik
        Pm[k]        = Pg_solved[i] / baseMVA

    E_mag  = np.abs(E_complex)
    delta0 = np.angle(E_complex)

    Yr_fault = _build_Yred(buses, branches, stab_gens, Vm, Va,
                            fault_bus=fault_bus, baseMVA=baseMVA)
    Pe_fault = _Pe(E_mag, delta0, Yr_fault)

    # dΔω/dt = (ωs / 2H) × (Pm − Pe)  →  ROCOF [Hz/s] = dΔω/dt / (2π)
    omega_s  = 2.0 * np.pi * 60.0
    accel    = (omega_s / (2.0 * H_eff)) * (Pm - Pe_fault)   # rad/s²
    rocof_hz = accel / (2.0 * np.pi)                          # Hz/s per generator

    # System ROCOF: inertia-weighted average of |ROCOF|
    rocof_sys = float(np.sum(H_eff * np.abs(rocof_hz)) / H_eff.sum())

    return dict(
        rocof_per_gen=rocof_hz,
        rocof_sys=rocof_sys,
        Pe_fault=Pe_fault,
        Pm=Pm,
    )


def sweep_cct_vs_penetration(
    buses, branches, stab_gens: list[StabGen],
    Vm: np.ndarray, Va: np.ndarray,
    Pg_solved: np.ndarray, Qg_solved: np.ndarray,
    fault_bus: int,
    drop_branch_idx: int | None = None,
    penetration_levels: list[float] | None = None,
    H_virtual: float = 3.0,
    t_lo: float = 0.02,
    t_hi: float = 1.5,
    tol: float = 0.02,
    t_end: float = 5.0,
    baseMVA: float = 100.0,
    progress_cb=None,
) -> dict:
    """
    Compute CCT vs DER penetration for Legacy and IEEE 1547-2018 models.

    Returns dict with keys 'penetration', 'cct_legacy', 'cct_1547',
    'H_sys_legacy', 'H_sys_1547', 'H_virtual', 'fault_bus'.
    """
    if penetration_levels is None:
        penetration_levels = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

    cct_legacy, cct_1547     = [], []
    H_sys_legacy, H_sys_1547 = [], []
    n_total = len(penetration_levels) * 2

    for step, pen in enumerate(penetration_levels):
        for der_type, cct_list, H_list in [
            ("legacy",   cct_legacy, H_sys_legacy),
            ("1547-2018", cct_1547,  H_sys_1547),
        ]:
            H_eff = effective_H(stab_gens, pen, der_type, H_virtual)
            cct_r = find_cct(buses, branches, stab_gens, Vm, Va,
                              Pg_solved, Qg_solved, fault_bus,
                              drop_branch_idx=drop_branch_idx,
                              H_override=H_eff,
                              t_lo=t_lo, t_hi=t_hi, tol=tol,
                              t_end=t_end, baseMVA=baseMVA)
            cct_list.append(cct_r["cct"])
            H_list.append(float(H_eff.sum()))

            if progress_cb is not None:
                progress_cb((step * 2 + (1 if der_type == "1547-2018" else 0) + 1) / n_total)

    return dict(
        penetration=penetration_levels,
        cct_legacy=cct_legacy,
        cct_1547=cct_1547,
        H_sys_legacy=H_sys_legacy,
        H_sys_1547=H_sys_1547,
        H_virtual=H_virtual,
        fault_bus=fault_bus,
    )
