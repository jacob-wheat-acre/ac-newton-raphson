
# acpf_nr.py
# Complete: NR solver + IEEE 3/9-bus cases + CSV + PNG (rounded for readability)
# Includes PV-as-PQ warm-up for first few iterations to improve convergence on IEEE-9.

from dataclasses import dataclass
from enum import Enum, auto
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Data structures
# -------------------------
class BusType(Enum):
    SLACK = auto()
    PV = auto()
    PQ = auto()


@dataclass
class Bus:
    num: int
    type: BusType
    Pd: float = 0.0      # MW
    Qd: float = 0.0      # MVAr
    Vm: float = 1.0      # pu (SLACK/PV specified; PQ initial guess)
    Va: float = 0.0      # degrees (SLACK specified; others initial guess)
    Gsh: float = 0.0     # pu
    Bsh: float = 0.0     # pu


@dataclass
class Gen:
    bus: int
    Pg: float = 0.0      # MW (input display only; solved Pg is computed)
    Qg: float = 0.0      # MVAr (input display only)
    Qmin: float = -1e9   # MVAr
    Qmax: float =  1e9   # MVAr
    Vset: float = 1.0    # pu target for SLACK/PV


@dataclass
class Branch:
    fbus: int
    tbus: int
    r: float             # pu
    x: float             # pu
    b: float = 0.0       # pu total line charging susceptance
    tap: float = 1.0     # off-nominal tap magnitude (on "from" side)
    shift: float = 0.0   # degrees, series phase shift (positive = delay)


# -------------------------
# Network builders & injections
# -------------------------
def build_ybus(nb, branches, buses_by_num):
    """
    Build Ybus with π-model lines, bus shunts, and off-nominal taps on 'from' side.
    From-side shunt is scaled by 1/|a|^2 when tap != 1.0.
    """
    Y = np.zeros((nb, nb), dtype=complex)
    for br in branches:
        i = br.fbus - 1
        k = br.tbus - 1
        z = complex(br.r, br.x)
        y = 1 / z
        bsh = 1j * br.b / 2.0
        tap = br.tap if br.tap != 0 else 1.0
        a = tap * np.exp(1j * np.deg2rad(br.shift))
        a2 = a * np.conj(a)  # |a|^2

        # π-model with off-nominal tap on "from" side
        Y[i, i] += (y / a2) + (bsh / a2)   # scale from-side shunt by 1/|a|^2
        Y[k, k] += y + bsh
        Y[i, k] += -y / np.conj(a)
        Y[k, i] += -y / a

    # Bus shunts
    for bnum, b in buses_by_num.items():
        idx = bnum - 1
        Y[idx, idx] += complex(b.Gsh, b.Bsh)
    return Y


def power_injections(Y: np.ndarray, Vm: np.ndarray, Va: np.ndarray):
    """
    Return net bus injections P, Q (pu). Convention: S = P + jQ (Q > 0 inductive).
    """
    Va_rad = np.deg2rad(Va)
    V = Vm * np.exp(1j * Va_rad)
    I = Y @ V
    S = V * np.conj(I)
    P = S.real
    Q = -S.imag   # IMPORTANT: -Im(S) so Jacobian formulas (H/N/M/L) are consistent
    return P, Q


def dc_warm_start_angles(nb, branches, slack, Pspec_pu):
    """
    Very simple DC PF initializer for bus angles (radians).
    - Uses only series reactances (ignores shunts and line charging).
    - Off-nominal taps are treated as 1.0 in this initializer.
    """
    # Build B' (series-only) matrix
    Bp = np.zeros((nb, nb), dtype=float)
    for br in branches:
        i = br.fbus - 1
        k = br.tbus - 1
        if br.x == 0.0:
            continue
        b_series = -1.0 / br.x  # since y = 1/(j x) = -j/x
        Bp[i, i] -= b_series
        Bp[k, k] -= b_series
        Bp[i, k] += b_series
        Bp[k, i] += b_series

    # Solve B' * theta = Pspec for non-slack buses
    idx = [i for i in range(nb) if i != slack]
    Bpp = Bp[np.ix_(idx, idx)]
    Pvec = Pspec_pu[idx]
    theta_ns = np.linalg.lstsq(Bpp, Pvec, rcond=None)[0]

    theta = np.zeros(nb)
    theta[idx] = theta_ns
    theta[slack] = 0.0
    return theta  # radians


# -------------------------
# Newton–Raphson AC power flow
# -------------------------
def newton_raphson_pf(
    buses: list[Bus],
    gens: list[Gen],
    branches: list[Branch],
    baseMVA: float = 100.0,
    tol: float = 1e-6,
    max_it: int = 60,
    verbose: bool = True,
):
    nb = len(buses)
    buses_by_num = {b.num: b for b in buses}
    slack_idx = [i for i, b in enumerate(buses) if b.type == BusType.SLACK]
    if len(slack_idx) != 1:
        raise ValueError("There must be exactly one Slack bus")
    slack = slack_idx[0]

    # Aggregate generator inputs (per-bus, pu)
    Pg = np.zeros(nb)
    Qg = np.zeros(nb)
    Qmin = np.full(nb, -1e9)
    Qmax = np.full(nb,  1e9)
    Vset = np.array([b.Vm for b in buses], dtype=float)
    for g in gens:
        i = g.bus - 1
        Pg[i] += g.Pg / baseMVA
        Qg[i] += g.Qg / baseMVA
        Qmin[i] = min(Qmin[i], g.Qmin / baseMVA)
        Qmax[i] = max(Qmax[i], g.Qmax / baseMVA)
        Vset[i] = g.Vset

    Pd = np.array([b.Pd for b in buses]) / baseMVA
    Qd = np.array([b.Qd for b in buses]) / baseMVA

    # Specified net injections
    Pspec = Pg - Pd
    Qspec = Qg - Qd

    # State vectors
    Vm = np.array([b.Vm for b in buses], dtype=float)
    Va = np.array([b.Va for b in buses], dtype=float)
    Va[slack] = buses[slack].Va
    Vm[slack] = buses[slack].Vm

    # DC warm-start for angles (one-time)
    if np.allclose(Va, 0.0):
        theta0 = dc_warm_start_angles(nb, branches, slack, Pspec)
        Va = np.rad2deg(theta0)
        Va[slack] = buses[slack].Va  # keep slack reference

    PV = np.array([i for i, b in enumerate(buses) if b.type == BusType.PV], dtype=int)
    PQ = np.array([i for i, b in enumerate(buses) if b.type == BusType.PQ], dtype=int)

    Y = build_ybus(nb, branches, buses_by_num)
    G = Y.real
    B = Y.imag

    p_idx = np.array([i for i in range(nb) if i != slack], dtype=int)

    WARMUP_ITERS = 5  # treat PV as PQ for the first few iterations

    for it in range(1, max_it + 1):
        # Effective PV/PQ sets with warm-up
        if it <= WARMUP_ITERS:
            PV_eff = np.array([], dtype=int)             # No PV behavior during warm-up
            PQ_eff = np.sort(np.r_[PQ, PV])              # PV act like PQ temporarily
            pin_pv = False
        else:
            PV_eff = PV
            PQ_eff = PQ
            pin_pv = True

        # Pin PV magnitudes only when PV behavior is active; normalize angles
        if pin_pv and len(PV_eff) > 0:
            Vm[PV_eff] = Vset[PV_eff]
        Va = ((Va + 180.0) % 360.0) - 180.0
        Va[slack] = buses[slack].Va

        P, Q = power_injections(Y, Vm, Va)

        dP = Pspec[p_idx] - P[p_idx]
        dQ = Qspec[PQ_eff] - Q[PQ_eff]

        misP = np.linalg.norm(dP, np.inf)
        misQ = np.linalg.norm(dQ, np.inf) if len(PQ_eff) else 0.0
        maxmis = max(misP, misQ)

        if verbose:
            print(
                f"   it={it:02d}  |dP|_inf={misP:.3e}  |dQ|_inf={misQ:.3e}  "
                f"PV={list(PV)} PQ={list(PQ)}  (eff: PV={list(PV_eff)} PQ={list(PQ_eff)})"
            )

        if maxmis < tol:
            converged = True
            break

        # Jacobian
        Va_rad = np.deg2rad(Va)
        H = np.zeros((len(p_idx), len(p_idx)))
        N = np.zeros((len(p_idx), len(PQ_eff)))
        M = np.zeros((len(PQ_eff), len(p_idx)))
        L = np.zeros((len(PQ_eff), len(PQ_eff)))
        pmap = {bus_i: k for k, bus_i in enumerate(p_idx)}
        qmap = {bus_i: k for k, bus_i in enumerate(PQ_eff)}

        for i in range(nb):
            for k in range(nb):
                thik = Va_rad[i] - Va_rad[k]
                Gik = G[i, k]
                Bik = B[i, k]

                if i != slack and k != slack:
                    if i == k:
                        H[pmap[i], pmap[k]] = +Q[i] - (Vm[i] ** 2) * B[i, i]
                    else:
                        H[pmap[i], pmap[k]] = Vm[i] * Vm[k] * (
                            -Gik * np.sin(thik) + Bik * np.cos(thik)
                        )
                if i != slack and k in PQ_eff:
                    if i == k:
                        N[pmap[i], qmap[k]] = P[i] / Vm[i] + G[i, i] * Vm[i]
                    else:
                        N[pmap[i], qmap[k]] = Vm[i] * (
                            Gik * np.cos(thik) + Bik * np.sin(thik)
                        )
                if i in PQ_eff and k != slack:
                    if i == k:
                        M[qmap[i], pmap[k]] = P[i] - (Vm[i] ** 2) * G[i, i]
                    else:
                        M[qmap[i], pmap[k]] = -Vm[i] * Vm[k] * (
                            Gik * np.cos(thik) + Bik * np.sin(thik)
                        )
                if i in PQ_eff and k in PQ_eff:
                    if i == k:
                        L[qmap[i], qmap[k]] = -Q[i] / Vm[i] - B[i, i] * Vm[i]
                    else:
                        L[qmap[i], qmap[k]] = Vm[i] * (
                            Gik * np.sin(thik) - Bik * np.cos(thik)
                        )

        J = np.block([[H, N],
                      [M, L]])

        mismatch = np.r_[dP, dQ]
        dx = np.linalg.solve(J, mismatch)
        dtheta = dx[:len(p_idx)]
        dV     = dx[len(p_idx):]

        # Trust-region caps (safe, reasonably large to leave plateaus)
        dtheta = np.clip(dtheta, -np.deg2rad(15.0), np.deg2rad(15.0))   # ±15°
        dV     = np.clip(dV,     -0.15,             0.15)               # ±0.15 pu

        # Strict backtracking with relative acceptance; linear |V| trial update
        alpha_try = 1.0
        alpha_min = 1e-6
        mis_prev  = maxmis
        accepted  = False

        while True:
            Va_trial = Va.copy()
            Vm_trial = Vm.copy()

            # Trial angle update (non-slack)
            Va_trial[p_idx] += np.rad2deg(alpha_try * dtheta)

            # Linear magnitude update for PQ buses + clamp to a safe band for trial
            if len(PQ_eff) > 0:
                Vm_trial[PQ_eff] = Vm[PQ_eff] + alpha_try * dV
                Vm_trial[PQ_eff] = np.clip(Vm_trial[PQ_eff], 0.75, 1.25)

            # Pin PV magnitudes only if PV behavior is active this iteration
            if pin_pv and len(PV_eff) > 0:
                Vm_trial[PV_eff] = Vset[PV_eff]

            # Normalize and repin slack
            Va_trial = ((Va_trial + 180.0) % 360.0) - 180.0
            Va_trial[slack] = buses[slack].Va

            # Trial mismatch
            P_t, Q_t = power_injections(Y, Vm_trial, Va_trial)
            dP_t = Pspec[p_idx] - P_t[p_idx]
            dQ_t = Qspec[PQ_eff] - Q_t[PQ_eff]
            mis_new = max(np.linalg.norm(dP_t, np.inf),
                          np.linalg.norm(dQ_t, np.inf) if len(PQ_eff) else 0.0)

            # Accept if we get a relative gain OR if we are already very small in absolute terms
            if (mis_new <= mis_prev * (1.0 - 1e-8)) or (mis_new < 1e-6):
                Va = Va_trial
                Vm = Vm_trial
                maxmis = mis_new
                accepted = True
                break

            alpha_try *= 0.5
            if alpha_try < alpha_min:
                break  # fallback below

        if not accepted:
            # Fallback: small but meaningful step (safe due to trust region)
            tiny = 0.10
            Va[p_idx] += np.rad2deg(tiny * dtheta)
            if len(PQ_eff) > 0:
                Vm[PQ_eff] = np.clip(Vm[PQ_eff] + tiny * dV, 0.80, 1.20)
            if pin_pv and len(PV_eff) > 0:
                Vm[PV_eff] = Vset[PV_eff]
            Va = ((Va + 180.0) % 360.0) - 180.0
            Va[slack] = buses[slack].Va

        # PV Q-limit enforcement (only when PV behavior is active)
        if pin_pv and len(PV_eff) > 0:
            _, Q_now = power_injections(Y, Vm, Va)
            flipped = []
            for i in list(PV_eff):
                Qg_candidate = Q_now[i] + Qd[i]  # pu MVAr
                if Qg_candidate < Qmin[i] - 1e-8:
                    Qspec[i] = Qmin[i] - Qd[i]
                    flipped.append(i)
                elif Qg_candidate > Qmax[i] + 1e-8:
                    Qspec[i] = Qmax[i] - Qd[i]
                    flipped.append(i)
            if flipped:
                PV = np.array([k for k in PV if k not in flipped], dtype=int)
                PQ = np.sort(np.r_[PQ, flipped])

    else:
        converged = False

    results = {
        "Vm": Vm,
        "Va": Va,
        "iterations": it if 'it' in locals() else 0,
        "converged": (converged if 'converged' in locals() else False),
    }
    if not results["converged"]:
        raise RuntimeError("NR power flow did not converge within max iterations")
    return results


# -------------------------
# Post-processing: flows, tables, CSVs, PNG
# -------------------------
def compute_branch_flows(Y, Vm, Va, branches, baseMVA=100.0):
    Va_rad = np.deg2rad(Va)
    V = Vm * np.exp(1j * Va_rad)
    rows = []
    for br in branches:
        i = br.fbus - 1
        k = br.tbus - 1
        z = complex(br.r, br.x)
        y = 1 / z
        bsh = 1j * br.b / 2.0
        tap = br.tap if br.tap != 0 else 1.0
        shift_rad = np.deg2rad(br.shift)
        a = tap * np.exp(1j * shift_rad)

        Iik = (V[i]/a - V[k]) * y + V[i]/a * bsh   # i -> k
        Iki = (V[k] - V[i]/a) * y + V[k]    * bsh   # k -> i
        S_ik = V[i] * np.conj(Iik)
        S_ki = V[k] * np.conj(Iki)

        Pik, Qik = S_ik.real * baseMVA, S_ik.imag * baseMVA
        Pki, Qki = S_ki.real * baseMVA, S_ki.imag * baseMVA

        rows.append({
            "From": br.fbus, "To": br.tbus,
            "P_from (MW)": f"{Pik:.2f}", "Q_from (MVAr)": f"{Qik:.2f}",
            "P_to (MW)":   f"{Pki:.2f}", "Q_to (MVAr)":   f"{Qki:.2f}",
            "P_loss (MW)": f"{(Pik + Pki):.2f}", "Q_loss (MVAr)": f"{(Qik + Qki):.2f}",
            "R (pu)": f"{br.r:.3f}", "X (pu)": f"{br.x:.3f}", "B (pu)": f"{br.b:.3f}",
            "Tap": f"{br.tap:.3f}", "Shift (deg)": f"{br.shift:.2f}"
        })
    return pd.DataFrame(rows)


def build_bus_gen_branch_tables(buses, gens, branches, results, baseMVA=100.0):
    Vm = results["Vm"]; Va = results["Va"]

    # Bus table (rounded for display)
    bus_rows = []
    for idx, b in enumerate(buses):
        bus_rows.append({
            "Bus": b.num,
            "Type": b.type.name,
            "Vm (pu)": f"{Vm[idx]:.3f}",
            "Va (deg)": f"{Va[idx]:.2f}",
            "Pd (MW)": f"{b.Pd:.2f}", "Qd (MVAr)": f"{b.Qd:.2f}",
            "Gsh (pu)": f"{b.Gsh:.3f}", "Bsh (pu)": f"{b.Bsh:.3f}",
        })
    bus_df = pd.DataFrame(bus_rows).sort_values("Bus")

    # Gen table (solved Pg/Qg implied by injections + load)
    buses_by_num = {b.num: b for b in buses}
    Y = build_ybus(len(buses), branches, buses_by_num)
    Pinj_pu, Qinj_pu = power_injections(Y, Vm, Va)
    Pd_MW = np.array([b.Pd for b in buses])
    Qd_MVAr = np.array([b.Qd for b in buses])
    Pg_MW = Pinj_pu * baseMVA + Pd_MW
    Qg_MVAr = Qinj_pu * baseMVA + Qd_MVAr

    gen_rows = []
    for g in gens:
        i = g.bus - 1
        gen_rows.append({
            "Gen@Bus": g.bus,
            "Pg_solved (MW)": f"{Pg_MW[i]:.2f}",
            "Qg_solved (MVAr)": f"{Qg_MVAr[i]:.2f}",
            "Pg_input (MW)": f"{g.Pg:.2f}", "Qg_input (MVAr)": f"{g.Qg:.2f}",
            "Qmin (MVAr)": f"{g.Qmin:.2f}", "Qmax (MVAr)": f"{g.Qmax:.2f}",
            "Vset (pu)": f"{g.Vset:.3f}"
        })
    gen_df = pd.DataFrame(gen_rows).sort_values("Gen@Bus")

    # Branch param table (rounded)
    branch_rows = []
    for br in branches:
        branch_rows.append({
            "From": br.fbus, "To": br.tbus,
            "R (pu)": f"{br.r:.3f}", "X (pu)": f"{br.x:.3f}", "B (pu)": f"{br.b:.3f}",
            "Tap": f"{br.tap:.3f}", "Shift (deg)": f"{br.shift:.2f}"
        })
    branch_df = pd.DataFrame(branch_rows).sort_values(["From", "To"])

    return bus_df, gen_df, branch_df


def export_csvs(buses, gens, branches, results, baseMVA=100.0, prefix="snapshot"):
    bus_df, gen_df, branch_df = build_bus_gen_branch_tables(buses, gens, branches, results, baseMVA)
    bus_df.to_csv(f"{prefix}_buses.csv", index=False)
    gen_df.to_csv(f"{prefix}_generators.csv", index=False)
    branch_df.to_csv(f"{prefix}_branches.csv", index=False)
    print(f"Wrote CSVs: {prefix}_buses.csv, {prefix}_generators.csv, {prefix}_branches.csv")


def _layout_on_circle(n: int):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.cos(angles), np.sin(angles)


def draw_network_png(
    filename_png,
    buses, gens, branches, results,
    baseMVA=100.0,
    include_flows=True,
    v_limits=(0.95, 1.05),
):
    Vm = results["Vm"]; Va = results["Va"]
    bus_df, gen_df, branch_df = build_bus_gen_branch_tables(
        buses, gens, branches, results, baseMVA=baseMVA
    )
    flows_df = None
    if include_flows:
        Y = build_ybus(len(buses), branches, {b.num: b for b in buses})
        flows_df = compute_branch_flows(Y, Vm, Va, branches, baseMVA=baseMVA)

    plt.close("all")
    fig = plt.figure(figsize=(16, 10), constrained_layout=False)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1.0, 1.0])

    ax_net   = fig.add_subplot(gs[0, :])
    ax_bus   = fig.add_subplot(gs[1, 0])
    ax_gen   = fig.add_subplot(gs[1, 1])
    ax_br    = fig.add_subplot(gs[1, 2])
    ax_flow  = fig.add_subplot(gs[2, :]) if include_flows else None

    fig.suptitle("AC Power Flow Snapshot", fontsize=16, weight="bold")

    # Network diagram
    n = len(buses)
    x, y = _layout_on_circle(n)
    size_base = 800
    sizes = size_base * (0.6 + 0.4 * Vm / max(1.0, Vm.max()))
    color = Vm - 1.0
    sc = ax_net.scatter(x, y, s=sizes, c=color, cmap="coolwarm", edgecolor="k", zorder=3)
    cbar = plt.colorbar(sc, ax=ax_net, pad=0.01, fraction=0.03)
    cbar.set_label("|V| - 1.0 (pu)")

    for br in branches:
        i = br.fbus - 1; k = br.tbus - 1
        ax_net.plot([x[i], x[k]], [y[i], y[k]], color="gray", lw=1.5, zorder=1)

    vmin, vmax = v_limits
    violations = (Vm < vmin) | (Vm > vmax)
    for i, b in enumerate(buses):
        if violations[i]:
            ax_net.scatter([x[i]], [y[i]], s=sizes[i]*1.2, facecolors='none',
                           edgecolors='red', linewidths=2.0, zorder=4)
        ax_net.text(x[i], y[i] + 0.07, f"{b.num}", ha="center", va="bottom",
                    fontsize=11, weight="bold")
        ax_net.text(x[i], y[i] - 0.04, f"{Vm[i]:.3f} pu\n{Va[i]:.1f}°",
                    ha="center", va="top", fontsize=9)

    ax_net.set_title(f"Network Diagram (node size ∝ |V|, color ∝ |V|−1.0)  "
                     f"Voltage limits: [{vmin:.2f}, {vmax:.2f}] pu")
    ax_net.axis("off")

    # Tables (truncate to avoid overcrowding the PNG)
    def _table(ax, df, title, max_rows=16):
        ax.axis("off")
        df_show = df.copy()
        if len(df_show) > max_rows:
            df_show = df_show.head(max_rows)
        tbl = ax.table(cellText=df_show.values, colLabels=df_show.columns,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1.0, 1.1)
        ax.set_title(title, fontsize=12, pad=6)

    _table(ax_bus, bus_df,    "Buses (solved state + loads/shunts)")
    _table(ax_gen, gen_df,    "Generators (solved Pg/Qg & limits)")
    _table(ax_br,  branch_df, "Branches (parameters)")

    if include_flows and flows_df is not None:
        ax_flow.axis("off")
        flows_sorted = flows_df.sort_values("P_loss (MW)", ascending=False)
        topN = min(18, len(flows_sorted))
        _table(ax_flow, flows_sorted.head(topN), f"Branch Flows (top {topN} by MW loss)")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(filename_png, dpi=200)
    plt.close(fig)
    print(f"PNG snapshot written to: {filename_png}")


# -------------------------
# Built-in cases
# -------------------------
def case_3bus():
    buses = [
        Bus(1, BusType.SLACK, Vm=1.06, Va=0.0),
        Bus(2, BusType.PV, Pd=0.2, Qd=0.1, Vm=1.045),
        Bus(3, BusType.PQ, Pd=0.45, Qd=0.15),
    ]
    gens = [
        Gen(1, Pg=100.0, Vset=1.06),                    # Pg is display input
        Gen(2, Pg=40.0,  Vset=1.045, Qmin=-40.0, Qmax=50.0),
    ]
    branches = [
        Branch(1, 2, r=0.02, x=0.06, b=0.03),
        Branch(1, 3, r=0.08, x=0.24, b=0.025),
        Branch(2, 3, r=0.06, x=0.18, b=0.02),
    ]
    return buses, gens, branches


def ieee_9bus_case():
    """
    WSCC / IEEE 9-bus (100 MVA base).
    - Slack: bus 1 @ 1.04 pu
    - PVs:   buses 2, 3 @ 1.025 pu
    - Loads only at buses 5, 7, 9
    """
    buses = [
        Bus(1, BusType.SLACK, Vm=1.04, Va=0.0),
        Bus(2, BusType.PV,    Vm=1.025),
        Bus(3, BusType.PV,    Vm=1.025),
        Bus(4, BusType.PQ),
        Bus(5, BusType.PQ, Pd=90.0,  Qd=30.0),
        Bus(6, BusType.PQ),
        Bus(7, BusType.PQ, Pd=100.0, Qd=35.0),
        Bus(8, BusType.PQ),
        Bus(9, BusType.PQ, Pd=125.0, Qd=50.0),
    ]

    gens = [
        Gen(1, Pg=71.0,  Vset=1.04,   Qmin=-999.0, Qmax=999.0),  # Slack
        Gen(2, Pg=163.0, Vset=1.025,  Qmin=-999.0, Qmax=999.0),  # PV
        Gen(3, Pg=85.0,  Vset=1.025,  Qmin=-999.0, Qmax=999.0),  # PV
    ]

    branches = [
        Branch(1, 4, r=0.0000, x=0.0576, b=0.000),
        Branch(4, 5, r=0.0170, x=0.0920, b=0.158),
        Branch(5, 6, r=0.0390, x=0.1700, b=0.358),
        Branch(3, 6, r=0.0000, x=0.0586, b=0.000),
        Branch(6, 7, r=0.0119, x=0.1008, b=0.209),
        Branch(7, 8, r=0.0085, x=0.0720, b=0.149),
        Branch(8, 2, r=0.0000, x=0.0625, b=0.000),
        Branch(8, 9, r=0.0320, x=0.1610, b=0.306),
        Branch(9, 4, r=0.0100, x=0.0850, b=0.176),
    ]
    return buses, gens, branches


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Choose your case here: "3bus" or "9bus"
    CASE = "9bus"

    if CASE == "3bus":
        buses, gens, branches = case_3bus()
    elif CASE == "9bus":
        buses, gens, branches = ieee_9bus_case()
    else:
        raise ValueError("Unknown CASE. Use '3bus' or '9bus'.")

    # --- Homotopy on line charging b (0 -> 1) with reseeding ---
    orig_b = [br.b for br in branches]

    def set_b_scale(scale: float):
        for br, b0 in zip(branches, orig_b):
            br.b = b0 * scale

    def seed_from(res):
        # carry forward the last solution as the new initial guess
        for i, bus in enumerate(buses):
            if bus.type != BusType.SLACK:
                bus.Va = float(res["Va"][i])
            if bus.type == BusType.PQ:
                bus.Vm = float(res["Vm"][i])

    res = None
    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        set_b_scale(s)
        if res is not None:
            seed_from(res)
        print(f"\n--- Solving with line-charging scale b = {s:.2f} ---")
        res = newton_raphson_pf(
            buses, gens, branches,
            baseMVA=100.0, tol=1e-6, max_it=120, verbose=True
        )
            
        try:
            res = newton_raphson_pf(
                buses, gens, branches,
                baseMVA=100.0, tol=1e-6, max_it=120, verbose=True
            )
        except RuntimeError as e:
            print(f"Stage b={s:.2f} failed: {e}")
            # Optionally relax settings and retry once:
            try:
                res = newton_raphson_pf(
                    buses, gens, branches,
                    baseMVA=100.0, tol=5e-6, max_it=200, verbose=True
                )
            except RuntimeError:
                # move on; later stages may still converge
                res = None
                continue


    # Final solve at b=1.0 (already solved above; this is harmless if you want to keep it)
    res = newton_raphson_pf(buses, gens, branches, baseMVA=100.0, tol=1e-6, verbose=True)
    print("\nConverged:", res["converged"], "in", res["iterations"], "iterations")
    for i, b in enumerate(buses, start=1):
        print(f"Bus {i:2d}: V = {res['Vm'][i-1]:.4f} pu, angle = {res['Va'][i-1]:.3f}°")

    # Export CSV + PNG
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{CASE}_{stamp}"
    export_csvs(buses, gens, branches, res, baseMVA=100.0, prefix=prefix)
    draw_network_png(f"{prefix}.png", buses, gens, branches, res,
                     baseMVA=100.0, include_flows=True, v_limits=(0.95, 1.05))
