
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
    rate: float = 0.0    # MVA thermal rating (0 = unconstrained)


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
    Q = S.imag    # standard: Q > 0 means injecting reactive power into the network
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

        if maxmis < tol and pin_pv:  # don't declare converged while PV buses are in PQ warmup mode
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
                        H[pmap[i], pmap[k]] = -Q[i] - (Vm[i] ** 2) * B[i, i]
                    else:
                        H[pmap[i], pmap[k]] = Vm[i] * Vm[k] * (
                            Gik * np.sin(thik) - Bik * np.cos(thik)
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
                        L[qmap[i], qmap[k]] = Q[i] / Vm[i] - B[i, i] * Vm[i]
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
def case_ieee14():
    """
    IEEE 14-bus test case (100 MVA base).
    Five generators, 20 branches (3 are off-nominal transformers), shunt cap at bus 9.
    Reference solution (MATPOWER case14):
      Bus 1: 1.0600 pu / 0.00°    Bus 6: 1.0700 pu / -14.22°
      Bus 2: 1.0450 pu / -4.98°   Bus 9: 1.0559 pu / -14.94°
      Bus 3: 1.0100 pu / -12.72°  Bus 14: 1.0355 pu / -16.04°
    """
    buses = [
        Bus(1,  BusType.SLACK, Vm=1.060, Va=0.0),
        Bus(2,  BusType.PV,   Pd=21.7,  Qd=12.7, Vm=1.045),
        Bus(3,  BusType.PV,   Pd=94.2,  Qd=19.0, Vm=1.010),
        Bus(4,  BusType.PQ,   Pd=47.8,  Qd=-3.9),
        Bus(5,  BusType.PQ,   Pd=7.6,   Qd=1.6),
        Bus(6,  BusType.PV,   Pd=11.2,  Qd=7.5,  Vm=1.070),
        Bus(7,  BusType.PQ),
        Bus(8,  BusType.PV,   Vm=1.090),
        Bus(9,  BusType.PQ,   Pd=29.5,  Qd=16.6, Bsh=0.19),  # 19 MVAr cap bank
        Bus(10, BusType.PQ,   Pd=9.0,   Qd=5.8),
        Bus(11, BusType.PQ,   Pd=3.5,   Qd=1.8),
        Bus(12, BusType.PQ,   Pd=6.1,   Qd=1.6),
        Bus(13, BusType.PQ,   Pd=13.5,  Qd=5.8),
        Bus(14, BusType.PQ,   Pd=14.9,  Qd=5.0),
    ]
    gens = [
        Gen(1, Pg=232.4, Vset=1.060, Qmin=-999.0, Qmax=999.0),
        Gen(2, Pg=40.0,  Vset=1.045, Qmin=-40.0,  Qmax=50.0),
        Gen(3, Pg=0.0,   Vset=1.010, Qmin=0.0,    Qmax=40.0),
        Gen(6, Pg=0.0,   Vset=1.070, Qmin=-6.0,   Qmax=24.0),
        Gen(8, Pg=0.0,   Vset=1.090, Qmin=-6.0,   Qmax=24.0),
    ]
    branches = [
        Branch(1,  2,  r=0.01938, x=0.05917, b=0.0528, rate=200),
        Branch(1,  5,  r=0.05403, x=0.22304, b=0.0492, rate=200),
        Branch(2,  3,  r=0.04699, x=0.19797, b=0.0438, rate=200),
        Branch(2,  4,  r=0.05811, x=0.17632, b=0.0374, rate=200),
        Branch(2,  5,  r=0.05695, x=0.17388, b=0.0340, rate=200),
        Branch(3,  4,  r=0.06701, x=0.17103, b=0.0346, rate=200),
        Branch(4,  5,  r=0.01335, x=0.04211, b=0.0128, rate=200),
        Branch(4,  7,  r=0.0,     x=0.20912, b=0.0,    tap=0.978, rate=350),  # transformer
        Branch(4,  9,  r=0.0,     x=0.55618, b=0.0,    tap=0.969, rate=350),  # transformer
        Branch(5,  6,  r=0.0,     x=0.25202, b=0.0,    tap=0.932, rate=350),  # transformer
        Branch(6,  11, r=0.09498, x=0.19890, b=0.0,    rate=200),
        Branch(6,  12, r=0.12291, x=0.25581, b=0.0,    rate=200),
        Branch(6,  13, r=0.06615, x=0.13027, b=0.0,    rate=200),
        Branch(7,  8,  r=0.0,     x=0.17615, b=0.0,    rate=200),
        Branch(7,  9,  r=0.0,     x=0.11001, b=0.0,    rate=200),
        Branch(9,  10, r=0.03181, x=0.08450, b=0.0,    rate=200),
        Branch(9,  14, r=0.12711, x=0.27038, b=0.0,    rate=200),
        Branch(10, 11, r=0.08205, x=0.19207, b=0.0,    rate=200),
        Branch(12, 13, r=0.22092, x=0.19988, b=0.0,    rate=200),
        Branch(13, 14, r=0.17093, x=0.34802, b=0.0,    rate=200),
    ]
    return buses, gens, branches


def case_ieee33():
    """
    IEEE 33-bus radial distribution feeder (Baran & Wu, 1989).
    12.66 kV, 100 MVA base.  All buses PQ; bus 1 is substation slack.
    Total load: 3.715 MW + j2.300 MVAr.
    Reference minimum voltage: ~0.9038 pu at bus 18 (Baran & Wu Fig. 4).
    """
    Zb = 1.6028  # Ω — Zbase = 12.66² / 100

    _br = [
        ( 1,  2, 0.0922, 0.0470), ( 2,  3, 0.4930, 0.2511), ( 3,  4, 0.3660, 0.1864),
        ( 4,  5, 0.3811, 0.1941), ( 5,  6, 0.8190, 0.7070), ( 6,  7, 0.1872, 0.6188),
        ( 7,  8, 0.7114, 0.2351), ( 8,  9, 1.0300, 0.7400), ( 9, 10, 1.0440, 0.7400),
        (10, 11, 0.1966, 0.0650), (11, 12, 0.3744, 0.1238), (12, 13, 1.4680, 1.1550),
        (13, 14, 0.5416, 0.7129), (14, 15, 0.5910, 0.5260), (15, 16, 0.7463, 0.5450),
        (16, 17, 1.2890, 1.7210), (17, 18, 0.7320, 0.5740),
        ( 2, 19, 0.1640, 0.1565), (19, 20, 1.5042, 1.3554), (20, 21, 0.4095, 0.4784),
        (21, 22, 0.7089, 0.9373),
        ( 3, 23, 0.4512, 0.3083), (23, 24, 0.8980, 0.7091), (24, 25, 0.8960, 0.7011),
        ( 6, 26, 0.2030, 0.1034), (26, 27, 0.2842, 0.1447), (27, 28, 1.0590, 0.9337),
        (28, 29, 0.8042, 0.7006), (29, 30, 0.5075, 0.2585), (30, 31, 0.9744, 0.9630),
        (31, 32, 0.3105, 0.3619), (32, 33, 0.3410, 0.5302),
    ]

    _ld = {                                              # bus: (kW, kVAr)
         2: (100,  60),  3: ( 90,  40),  4: (120,  80),  5: ( 60,  30),
         6: ( 60,  20),  7: (200, 100),  8: (200, 100),  9: ( 60,  20),
        10: ( 60,  20), 11: ( 45,  30), 12: ( 60,  35), 13: ( 60,  35),
        14: (120,  80), 15: ( 60,  10), 16: ( 60,  20), 17: ( 60,  20),
        18: ( 90,  40), 19: ( 90,  40), 20: ( 90,  40), 21: ( 90,  40),
        22: ( 90,  40), 23: ( 90,  50), 24: (420, 200), 25: (420, 200),
        26: ( 60,  25), 27: ( 60,  25), 28: ( 60,  20), 29: (120,  70),
        30: (200, 600), 31: (150,  70), 32: (210, 100), 33: ( 60,  40),
    }

    buses = [Bus(1, BusType.SLACK, Vm=1.0, Va=0.0)]
    for n in range(2, 34):
        kw, kvar = _ld.get(n, (0, 0))
        buses.append(Bus(n, BusType.PQ, Pd=kw / 1000.0, Qd=kvar / 1000.0))

    gens = [Gen(1, Pg=5.0, Vset=1.0)]   # substation source

    branches = [
        Branch(f, t, r=r_ohm / Zb, x=x_ohm / Zb, rate=10.0)
        for f, t, r_ohm, x_ohm in _br
    ]
    return buses, gens, branches


def case_ieee39():
    """
    IEEE 39-Bus New England test system (100 MVA base).
    10 generators; bus 31 is SLACK (external system equivalent).
    37 transmission lines + 9 step-up transformers = 46 branches.
    Data: Athay, Podmore & Virmani (1979); Anderson & Fouad (2nd ed.).
    All bus voltages are typically within 0.97–1.07 pu in the base case.
    """
    buses = [
        Bus( 1, BusType.PQ),
        Bus( 2, BusType.PQ),
        Bus( 3, BusType.PQ,   Pd= 322.0, Qd=   2.4),
        Bus( 4, BusType.PQ,   Pd= 500.0, Qd= 184.0),
        Bus( 5, BusType.PQ),
        Bus( 6, BusType.PQ),
        Bus( 7, BusType.PQ,   Pd= 233.8, Qd=  84.0),
        Bus( 8, BusType.PQ,   Pd= 522.0, Qd= 176.6),
        Bus( 9, BusType.PQ),
        Bus(10, BusType.PQ),
        Bus(11, BusType.PQ),
        Bus(12, BusType.PQ,   Pd=   8.5, Qd= -88.0),   # shunt capacitor
        Bus(13, BusType.PQ),
        Bus(14, BusType.PQ),
        Bus(15, BusType.PQ,   Pd= 320.0, Qd= 153.0),
        Bus(16, BusType.PQ,   Pd= 329.0, Qd=  32.3),
        Bus(17, BusType.PQ),
        Bus(18, BusType.PQ,   Pd= 158.0, Qd=  30.0),
        Bus(19, BusType.PQ),
        Bus(20, BusType.PQ,   Pd= 680.0, Qd= 103.0),
        Bus(21, BusType.PQ,   Pd= 274.0, Qd= 115.0),
        Bus(22, BusType.PQ),
        Bus(23, BusType.PQ,   Pd= 247.5, Qd=  84.6),
        Bus(24, BusType.PQ,   Pd= 308.6, Qd= -92.2),   # shunt capacitor
        Bus(25, BusType.PQ,   Pd= 224.0, Qd=  47.2),
        Bus(26, BusType.PQ,   Pd= 139.0, Qd=  17.0),
        Bus(27, BusType.PQ,   Pd= 281.0, Qd=  75.5),
        Bus(28, BusType.PQ,   Pd= 206.0, Qd=  27.6),
        Bus(29, BusType.PQ,   Pd= 283.5, Qd=  26.9),
        Bus(30, BusType.PV,   Vm=1.0475),
        Bus(31, BusType.SLACK, Pd=9.2,   Qd=4.6, Vm=0.982, Va=0.0),
        Bus(32, BusType.PV,   Vm=0.9831),
        Bus(33, BusType.PV,   Vm=0.9972),
        Bus(34, BusType.PV,   Vm=1.0123),
        Bus(35, BusType.PV,   Vm=1.0493),
        Bus(36, BusType.PV,   Vm=1.0635),
        Bus(37, BusType.PV,   Vm=1.0278),
        Bus(38, BusType.PV,   Vm=1.0265),
        Bus(39, BusType.PV,   Pd=1104.0, Qd=250.0, Vm=1.03),
    ]
    gens = [
        Gen(30, Pg= 250.0, Vset=1.0475),
        Gen(31, Pg= 677.0, Vset=0.9820),   # slack — Pg not used in solve
        Gen(32, Pg= 650.0, Vset=0.9831),
        Gen(33, Pg= 632.0, Vset=0.9972),
        Gen(34, Pg= 508.0, Vset=1.0123),
        Gen(35, Pg= 650.0, Vset=1.0493),
        Gen(36, Pg= 560.0, Vset=1.0635),
        Gen(37, Pg= 540.0, Vset=1.0278),
        Gen(38, Pg= 830.0, Vset=1.0265),
        Gen(39, Pg=1000.0, Vset=1.0300),
    ]
    branches = [
        # Transmission lines (b = total shunt charging susceptance)
        Branch( 1,  2, r=0.0035, x=0.0411, b=0.6987, rate=600),
        Branch( 1, 39, r=0.0010, x=0.0250, b=0.7500, rate=1000),
        Branch( 2,  3, r=0.0013, x=0.0151, b=0.2572, rate=500),
        Branch( 2, 25, r=0.0070, x=0.0086, b=0.1460, rate=500),
        Branch( 3,  4, r=0.0013, x=0.0213, b=0.2214, rate=500),
        Branch( 3, 18, r=0.0011, x=0.0133, b=0.2138, rate=500),
        Branch( 4,  5, r=0.0008, x=0.0128, b=0.1342, rate=600),
        Branch( 4, 14, r=0.0008, x=0.0129, b=0.1382, rate=500),
        Branch( 5,  6, r=0.0002, x=0.0026, b=0.0434, rate=1200),
        Branch( 5,  8, r=0.0008, x=0.0112, b=0.1476, rate=900),
        Branch( 6,  7, r=0.0006, x=0.0092, b=0.1130, rate=900),
        Branch( 6, 11, r=0.0007, x=0.0082, b=0.1389, rate=480),
        Branch( 7,  8, r=0.0004, x=0.0046, b=0.0780, rate=900),
        Branch( 8,  9, r=0.0023, x=0.0363, b=0.3804, rate=900),
        Branch( 9, 39, r=0.0010, x=0.0250, b=1.2000, rate=900),
        Branch(10, 11, r=0.0004, x=0.0043, b=0.0729, rate=600),
        Branch(10, 13, r=0.0004, x=0.0043, b=0.0729, rate=600),
        Branch(11, 12, r=0.0016, x=0.0435, b=0.0,    rate=500),
        Branch(12, 13, r=0.0016, x=0.0435, b=0.0,    rate=500),
        Branch(13, 14, r=0.0009, x=0.0101, b=0.1723, rate=500),
        Branch(14, 15, r=0.0018, x=0.0217, b=0.3660, rate=500),
        Branch(15, 16, r=0.0009, x=0.0094, b=0.1710, rate=500),
        Branch(16, 17, r=0.0007, x=0.0089, b=0.1342, rate=500),
        Branch(16, 19, r=0.0016, x=0.0195, b=0.3040, rate=500),
        Branch(16, 21, r=0.0008, x=0.0135, b=0.2548, rate=500),
        Branch(16, 24, r=0.0003, x=0.0059, b=0.0680, rate=500),
        Branch(17, 18, r=0.0007, x=0.0082, b=0.1319, rate=500),
        Branch(17, 27, r=0.0013, x=0.0173, b=0.3216, rate=500),
        Branch(19, 20, r=0.0007, x=0.0138, b=0.0,    rate=900),
        Branch(21, 22, r=0.0008, x=0.0140, b=0.2565, rate=900),
        Branch(22, 23, r=0.0006, x=0.0096, b=0.1846, rate=600),
        Branch(23, 24, r=0.0022, x=0.0350, b=0.3610, rate=600),
        Branch(25, 26, r=0.0032, x=0.0323, b=0.5130, rate=600),
        Branch(26, 27, r=0.0014, x=0.0147, b=0.2396, rate=600),
        Branch(26, 28, r=0.0043, x=0.0474, b=0.7802, rate=600),
        Branch(26, 29, r=0.0057, x=0.0625, b=1.0290, rate=600),
        Branch(28, 29, r=0.0014, x=0.0151, b=0.2490, rate=600),
        # Step-up transformers (tap on from-bus / HV side)
        Branch( 2, 30, r=0.0, x=0.0181, b=0.0, tap=1.025, rate=900),
        Branch( 6, 31, r=0.0, x=0.0250, b=0.0, tap=1.070, rate=900),
        Branch(10, 32, r=0.0, x=0.0200, b=0.0, tap=1.070, rate=900),
        Branch(19, 33, r=0.0, x=0.0142, b=0.0, tap=1.070, rate=900),
        Branch(20, 34, r=0.0, x=0.0180, b=0.0, tap=1.009, rate=900),
        Branch(22, 35, r=0.0, x=0.0143, b=0.0, tap=1.025, rate=900),
        Branch(23, 36, r=0.0, x=0.0272, b=0.0,            rate=900),  # nominal tap
        Branch(25, 37, r=0.0, x=0.0232, b=0.0, tap=1.025, rate=900),
        Branch(29, 38, r=0.0, x=0.0156, b=0.0, tap=1.025, rate=900),
    ]
    return buses, gens, branches


def case_ieee69():
    """
    IEEE 69-bus radial distribution test feeder (Das, Kothari & Kalam, 1995).
    12.66 kV, 100 MVA base.  All buses PQ; bus 1 is substation slack.
    Topology: main trunk 1→27, lateral from bus 3 (buses 28→65, folded),
              short lateral from bus 11 (buses 66→69).
    Total load: ~3.80 MW + j2.69 MVAr.
    Note: The long lateral has high-resistance sections (branches 33-34, 34-35)
    combined with a large concentrated load near bus 61, producing a severe
    undervoltage at bus 65 (~0.80 pu). This illustrates the motivation for
    reactive compensation and feeder reconfiguration in radial distribution.
    """
    Zb = 1.6028  # Ω — Zbase = 12.66² / 100 MVA

    # (from, to, R_Ω, X_Ω, P_kW at to-bus, Q_kVAr at to-bus)
    _data = [
        # Main trunk 1→27
        ( 1,  2, 0.0005, 0.0012,    0.0,   0.0),
        ( 2,  3, 0.0005, 0.0012,    0.0,   0.0),
        ( 3,  4, 0.0015, 0.0036,    0.0,   0.0),
        ( 4,  5, 0.0251, 0.0294,    0.0,   0.0),
        ( 5,  6, 0.3660, 0.1864,    2.6,   2.2),
        ( 6,  7, 0.3811, 0.1941,   40.4,  30.0),
        ( 7,  8, 0.0922, 0.0470,   75.0,  54.0),
        ( 8,  9, 0.0493, 0.0251,   30.0,  22.0),
        ( 9, 10, 0.8190, 0.2707,   28.0,  19.0),
        (10, 11, 0.1872, 0.0619,  145.0, 104.0),
        (11, 12, 0.7114, 0.2351,  145.0, 104.0),
        (12, 13, 1.0300, 0.3400,    8.0,   5.5),
        (13, 14, 1.0440, 0.3450,    8.0,   5.5),
        (14, 15, 1.0580, 0.3496,    0.0,   0.0),
        (15, 16, 0.1966, 0.0650,   45.5,  30.0),
        (16, 17, 0.3744, 0.1238,   60.0,  35.0),
        (17, 18, 0.0047, 0.0016,   60.0,  35.0),
        (18, 19, 0.3276, 0.1083,    0.0,   0.0),
        (19, 20, 0.2106, 0.0690,    1.0,   0.6),
        (20, 21, 0.3416, 0.1129,  114.0,  81.0),
        (21, 22, 0.0140, 0.0046,    5.0,   3.5),
        (22, 23, 0.1591, 0.0526,    0.0,   0.0),
        (23, 24, 0.3463, 0.1145,   28.0,  20.0),
        (24, 25, 0.7488, 0.2475,    0.0,   0.0),
        (25, 26, 0.3089, 0.1021,   14.0,  10.0),
        (26, 27, 0.1732, 0.0572,   14.0,  10.0),
        # Long lateral from bus 3 (buses 28→47, then fold to buses 48→65)
        ( 3, 28, 0.0044, 0.0108,   26.0,  18.6),
        (28, 29, 0.0640, 0.1565,   26.0,  18.6),
        (29, 30, 0.3978, 0.1315,    0.0,   0.0),
        (30, 31, 0.0702, 0.0232,    0.0,   0.0),
        (31, 32, 0.3510, 0.1160,    0.0,   0.0),
        (32, 33, 0.8390, 0.2816,    0.0,   0.0),
        (33, 34, 1.7080, 0.5646,    0.0,   0.0),
        (34, 35, 1.4740, 0.4873,    0.0,   0.0),
        (35, 36, 0.0966, 0.0319,   26.0,  18.6),
        (36, 37, 0.1230, 0.0406,   26.0,  18.6),
        (37, 38, 0.0034, 0.0011,    0.0,   0.0),
        (38, 39, 0.0851, 0.0281,   24.0,  17.2),
        (39, 40, 0.2898, 0.0958,   24.0,  17.2),
        (40, 41, 0.0822, 0.0272,    1.2,   1.0),
        (41, 42, 0.0928, 0.0306,    0.0,   0.0),
        (42, 43, 0.0434, 0.0143,    6.0,   4.3),
        (43, 44, 0.0608, 0.0201,    0.0,   0.0),
        (44, 45, 0.0608, 0.0201,   39.22, 26.3),
        (45, 46, 0.0608, 0.0201,   39.22, 26.3),
        (46, 47, 0.0608, 0.0201,    0.0,   0.0),
        (47, 48, 0.0997, 0.0330,   79.0,  56.4),
        (48, 49, 0.0367, 0.0121,  384.7, 274.5),
        (49, 50, 0.0735, 0.0243,  384.7, 274.5),
        (50, 51, 0.0735, 0.0243,   40.5,  28.3),
        (51, 52, 0.0735, 0.0243,    3.6,   2.7),
        (52, 53, 0.1692, 0.0559,    4.35,  3.5),
        (53, 54, 0.0010, 0.0010,   26.4,  19.0),
        (54, 55, 0.1401, 0.0463,   24.0,  17.2),
        (55, 56, 0.1401, 0.0463,    0.0,   0.0),
        (56, 57, 0.1253, 0.0414,    0.0,   0.0),
        (57, 58, 0.1253, 0.0414,    0.0,   0.0),
        (58, 59, 0.5764, 0.1906,  100.0,  72.0),
        (59, 60, 0.5764, 0.1906,    0.0,   0.0),
        (60, 61, 0.1254, 0.0414, 1244.0, 888.0),
        (61, 62, 0.1254, 0.0414,   32.0,  23.0),
        (62, 63, 0.1254, 0.0414,    0.0,   0.0),
        (63, 64, 0.1268, 0.0419,  227.0, 162.0),
        (64, 65, 0.1268, 0.0419,   59.0,  42.0),
        # Short lateral from bus 11 (buses 66→69)
        (11, 66, 0.0044, 0.0108,   18.0,  13.0),
        (66, 67, 0.0640, 0.1565,   18.0,  13.0),
        (67, 68, 0.1053, 0.1230,   28.0,  20.0),
        (68, 69, 0.0304, 0.0355,   28.0,  20.0),
    ]

    load_at: dict[int, tuple[float, float]] = {}
    for f, t, r, x, p, q in _data:
        if p > 0 or q > 0:
            load_at[t] = (p, q)

    buses = [Bus(1, BusType.SLACK, Vm=1.0, Va=0.0)]
    for n in range(2, 70):
        p, q = load_at.get(n, (0.0, 0.0))
        buses.append(Bus(n, BusType.PQ, Pd=p / 1000.0, Qd=q / 1000.0))

    gens = [Gen(1, Pg=10.0, Vset=1.0)]

    branches = [
        Branch(f, t, r=r_ohm / Zb, x=x_ohm / Zb, rate=10.0)
        for f, t, r_ohm, x_ohm, p, q in _data
    ]
    return buses, gens, branches


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
        Branch(1, 2, r=0.02, x=0.06, b=0.03,  rate=100),
        Branch(1, 3, r=0.08, x=0.24, b=0.025, rate=100),
        Branch(2, 3, r=0.06, x=0.18, b=0.02,  rate=100),
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
        Branch(1, 4, r=0.0000, x=0.0576, b=0.000, rate=250),
        Branch(4, 5, r=0.0170, x=0.0920, b=0.158, rate=250),
        Branch(5, 6, r=0.0390, x=0.1700, b=0.358, rate=150),
        Branch(3, 6, r=0.0000, x=0.0586, b=0.000, rate=300),
        Branch(6, 7, r=0.0119, x=0.1008, b=0.209, rate=150),
        Branch(7, 8, r=0.0085, x=0.0720, b=0.149, rate=250),
        Branch(8, 2, r=0.0000, x=0.0625, b=0.000, rate=250),
        Branch(8, 9, r=0.0320, x=0.1610, b=0.306, rate=250),
        Branch(9, 4, r=0.0100, x=0.0850, b=0.176, rate=300),
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
        try:
            res = newton_raphson_pf(
                buses, gens, branches,
                baseMVA=100.0, tol=1e-6, max_it=60, verbose=True
            )
        except RuntimeError as e:
            print(f"Stage b={s:.2f} failed: {e}")
            res = None
            continue

    if res is None:
        raise RuntimeError("Homotopy failed to converge at b=1.0")
    print("\nConverged:", res["converged"], "in", res["iterations"], "iterations")
    for i, b in enumerate(buses, start=1):
        print(f"Bus {i:2d}: V = {res['Vm'][i-1]:.4f} pu, angle = {res['Va'][i-1]:.3f}°")

    # Export CSV + PNG
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{CASE}_{stamp}"
    export_csvs(buses, gens, branches, res, baseMVA=100.0, prefix=prefix)
    draw_network_png(f"{prefix}.png", buses, gens, branches, res,
                     baseMVA=100.0, include_flows=True, v_limits=(0.95, 1.05))
