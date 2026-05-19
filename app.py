import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import copy
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from acpf_nr import (
    Bus, BusType, Gen, Branch,
    newton_raphson_pf, build_ybus, power_injections,
    compute_branch_flows, case_3bus, ieee_9bus_case, case_ieee14,
    case_ieee33, case_ieee39, case_ieee69,
)
from opf import dc_opf, case_opf_gens
from stability import (default_stab_gens, run_stability, find_cct,
                       effective_H, compute_rocof, sweep_cct_vs_penetration)

st.set_page_config(page_title="AC Power Flow", page_icon="⚡", layout="wide")
st.title("⚡ AC Newton–Raphson Power Flow")
st.markdown(
    "Adjust loads and generator voltage targets in the sidebar. "
    "The solver re-runs automatically and the diagram updates in real time."
)
st.divider()

# ── Bus type colors ────────────────────────────────────────────────────────────
C_SLACK    = "#FFD700"   # gold
C_PV       = "#90EE90"   # light green
C_PQ_LOAD  = "#FFA07A"   # light salmon
C_PQ_XFER  = "#B0C4DE"   # light steel blue (transit / no load)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("System")
    sys_type = st.radio("System type", ["Transmission", "Distribution"], horizontal=True)
    if sys_type == "Transmission":
        case_name = st.selectbox(
            "Case",
            ["3-Bus Example", "IEEE 9-Bus (WSCC)", "IEEE 14-Bus",
             "IEEE 39-Bus (New England)"],
        )
    else:
        case_name = st.selectbox(
            "Case",
            ["IEEE 33-Bus (Baran & Wu)", "IEEE 69-Bus (Das et al.)"],
        )

    if "69" in case_name:
        buses0, gens0, branches0 = case_ieee69()
    elif "39" in case_name:
        buses0, gens0, branches0 = case_ieee39()
    elif "33" in case_name:
        buses0, gens0, branches0 = case_ieee33()
    elif "14" in case_name:
        buses0, gens0, branches0 = case_ieee14()
    elif "9" in case_name:
        buses0, gens0, branches0 = ieee_9bus_case()
    else:
        buses0, gens0, branches0 = case_3bus()

    st.subheader("Generator voltage setpoints")
    vset_inputs: dict[int, float] = {}
    for g in gens0:
        b0 = next(b for b in buses0 if b.num == g.bus)
        tag = "Slack" if b0.type == BusType.SLACK else "PV"
        vset_inputs[g.bus] = st.slider(
            f"Bus {g.bus} ({tag})", 0.95, 1.10, float(g.Vset), 0.005,
            format="%.3f pu",
        )

    st.subheader("Load levels")
    load_inputs: dict[int, tuple[float, float]] = {}
    load_scale: float = 1.0
    _large_case = any(k in case_name for k in ("33", "39", "69"))
    if _large_case:
        _n = {"33": 32, "39": 19, "69": 68}.get(
            next(k for k in ("69", "39", "33") if k in case_name), 0
        )
        load_scale = st.slider(
            "Load scaling factor", 0.5, 2.0, 1.0, 0.05,
            help=f"Scales all {_n} load buses uniformly",
        )
    else:
        for b0 in buses0:
            if b0.Pd == 0 and b0.Qd == 0:
                continue
            st.markdown(f"**Bus {b0.num}**")
            pd_max  = max(b0.Pd * 3.0, 10.0)
            qd_max  = max(b0.Qd * 3.0,  5.0)
            step_pd = max(round(b0.Pd * 0.05, 1), 0.5)
            step_qd = max(round(b0.Qd * 0.05, 1), 0.5)
            pd_val = st.slider(f"Pd — bus {b0.num} (MW)",   0.0, pd_max, float(b0.Pd), step_pd)
            qd_val = st.slider(f"Qd — bus {b0.num} (MVAr)", 0.0, qd_max, float(b0.Qd), step_qd)
            load_inputs[b0.num] = (pd_val, qd_val)

    st.subheader("Solver")
    tol    = st.select_slider("Tolerance", [1e-4, 1e-5, 1e-6, 1e-8], value=1e-6,
                               format_func=lambda v: f"{v:.0e}")
    max_it = st.slider("Max iterations", 20, 200, 60)

    st.subheader("Display")
    label_size = st.slider(
        "Label size", 0.5, 3.0, 1.0, 0.1,
        help="Scale all bus number and voltage labels — increase when zoomed in",
    )

# ── Apply user changes to a fresh copy of the case ────────────────────────────
buses    = copy.deepcopy(buses0)
gens     = copy.deepcopy(gens0)
branches = copy.deepcopy(branches0)

for b in buses:
    if b.num in load_inputs:
        b.Pd, b.Qd = load_inputs[b.num]
    elif load_scale != 1.0:
        b.Pd *= load_scale
        b.Qd *= load_scale

for g in gens:
    if g.bus in vset_inputs:
        g.Vset = vset_inputs[g.bus]
for b in buses:
    for g in gens:
        if b.num == g.bus:
            b.Vm = g.Vset

# ── Solve with line-charging homotopy for robustness ──────────────────────────
orig_b = [br.b for br in branches]

def _set_b(scale: float) -> None:
    for br, b0v in zip(branches, orig_b):
        br.b = b0v * scale

def _seed(res: dict) -> None:
    for i, bus in enumerate(buses):
        if bus.type != BusType.SLACK:
            bus.Va = float(res["Va"][i])
        if bus.type == BusType.PQ:
            bus.Vm = float(res["Vm"][i])

results: dict | None = None
error_msg: str = ""

for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
    _set_b(s)
    if results is not None:
        _seed(results)
    try:
        results = newton_raphson_pf(
            buses, gens, branches,
            baseMVA=100.0, tol=tol, max_it=max_it, verbose=False,
        )
    except RuntimeError as e:
        error_msg = str(e)
        results = None
        break

if results is None:
    st.error(f"Power flow did not converge. {error_msg}")
    st.info("Try reducing the load or checking that the system is connected.")
    st.stop()

st.success(f"Converged in {results['iterations']} iterations.")

# ── Post-process ───────────────────────────────────────────────────────────────
Vm  = results["Vm"]
Va  = results["Va"]
nb  = len(buses)

Y            = build_ybus(nb, branches, {b.num: b for b in buses})
Pinj, Qinj   = power_injections(Y, Vm, Va)
Pd_arr       = np.array([b.Pd for b in buses])
Qd_arr       = np.array([b.Qd for b in buses])
Pg_solved    = Pinj * 100.0 + Pd_arr   # MW
Qg_solved    = Qinj * 100.0 + Qd_arr   # MVAr

flows        = compute_branch_flows(Y, Vm, Va, branches, baseMVA=100.0)
P_from       = [float(flows["P_from (MW)"].iloc[j])   for j in range(len(branches))]
Q_from       = [float(flows["Q_from (MVAr)"].iloc[j]) for j in range(len(branches))]
P_loss       = [float(flows["P_loss (MW)"].iloc[j])   for j in range(len(branches))]

V_LO, V_HI  = 0.95, 1.05
violations   = (Vm < V_LO) | (Vm > V_HI)

# ── Reference layouts for standard IEEE test cases ────────────────────────────
# Coordinates from published one-line diagrams (Anderson & Fouad, Glover & Sarma,
# Baran & Wu).  All other cases fall back to a circular layout.
CASE_COORDS: dict[str, dict[int, tuple[float, float]]] = {
    "3-Bus Example": {
        1: ( 0.0,  1.2),
        2: ( 1.2, -0.7),
        3: (-1.2, -0.7),
    },
    "IEEE 9-Bus (WSCC)": {
        # Generators at outer corners; HV ring 4-5-6-7-8-9-4 in the centre
        1: (-2.0,  1.5),   # Slack / Gen 1
        2: ( 2.0,  1.5),   # Gen 2
        3: (-2.0, -1.5),   # Gen 3
        4: (-1.0,  0.6),   # HV (xfmr from bus 1)
        5: (-1.6, -0.4),   # Load
        6: (-0.5, -0.4),   # HV (xfmr from bus 3)
        7: ( 0.5, -0.4),   # Load
        8: ( 1.0,  0.6),   # HV (xfmr from bus 2)
        9: ( 0.0,  0.6),   # Load
    },
    "IEEE 14-Bus": {
        1:  (-3.5,  0.0),   # Slack/Gen 1
        2:  (-2.0,  1.2),   # Gen 2
        3:  (-0.5,  2.5),   # Sync condenser
        4:  (-0.5,  0.5),   # Central transit hub
        5:  (-2.0, -1.2),   # Load
        6:  ( 1.5,  0.5),   # Gen 6
        7:  ( 0.5,  0.5),   # Transit (xfmrs to 4, 8, 9)
        8:  ( 0.5,  1.8),   # Sync condenser (xfmr to 7)
        9:  ( 2.5,  0.0),   # Load
        10: ( 3.5,  0.0),   # Load
        11: ( 3.5,  0.5),   # Load
        12: ( 4.5,  1.5),   # Load
        13: ( 4.5,  0.5),   # Load
        14: ( 3.5, -0.5),   # Load
    },
    "IEEE 33-Bus (Baran & Wu)": {
        # Main trunk (1→18): single horizontal row across the top
        1: (-2.00, 1.6),  2: (-1.76, 1.6),  3: (-1.52, 1.6),  4: (-1.28, 1.6),
        5: (-1.04, 1.6),  6: (-0.80, 1.6),  7: (-0.56, 1.6),  8: (-0.32, 1.6),
        9: (-0.08, 1.6), 10: ( 0.16, 1.6), 11: ( 0.40, 1.6), 12: ( 0.64, 1.6),
        13: ( 0.88, 1.6), 14: ( 1.12, 1.6), 15: ( 1.36, 1.6), 16: ( 1.60, 1.6),
        17: ( 1.84, 1.6), 18: ( 2.08, 1.6),
        # Lateral A (from bus 2): drops down at x = -1.76
        19: (-1.76, 0.8), 20: (-1.76, 0.0), 21: (-1.76, -0.8), 22: (-1.76, -1.6),
        # Lateral B (from bus 3): drops down at x = -1.52
        23: (-1.52, 0.8), 24: (-1.52, 0.0), 25: (-1.52, -0.8),
        # Lateral C (from bus 6): drops to bus 28, then runs right to bus 33
        26: (-0.80, 0.8), 27: (-0.80, 0.0), 28: (-0.80, -0.8),
        29: (-0.32, -0.8), 30: ( 0.16, -0.8), 31: ( 0.64, -0.8),
        32: ( 1.12, -0.8), 33: ( 1.60, -0.8),
    },
    "IEEE 39-Bus (New England)": {
        # Outer transmission ring (1→2→3→4→5→6→7→8→9→39→1, chord 5→8)
        1:  (-2.0,  2.0),  2:  (-0.5,  2.5),  3:  ( 1.2,  2.5),
        4:  ( 2.5,  1.5),  5:  ( 2.5,  0.0),  6:  ( 2.5, -1.5),
        7:  ( 1.5, -2.5),  8:  ( 0.0, -2.5),  9:  (-1.5, -2.5),
        39: (-2.5,  0.0),
        # Inner mesh
        10: ( 1.8,  0.5), 11: ( 1.8, -0.5), 12: ( 0.8, -0.5),
        13: ( 1.2,  0.5), 14: ( 1.2,  1.5), 15: ( 0.5,  0.8),
        16: ( 0.0,  0.5), 17: (-0.5,  1.0), 18: ( 0.5,  2.0),
        19: ( 0.0, -0.5), 20: (-0.5, -1.5), 21: ( 0.5, -0.5),
        22: ( 1.0, -1.0), 23: ( 1.5, -1.5), 24: ( 0.5, -1.5),
        25: (-1.2,  1.5), 26: (-1.5,  0.5), 27: (-1.0,  0.5),
        28: (-1.5, -0.5), 29: (-1.0, -1.0),
        # Generator terminal buses (step-up transformers to HV ring)
        30: (-0.5,  3.3),   # xfmr ← bus 2
        31: ( 3.3, -1.5),   # SLACK, xfmr ← bus 6
        32: ( 3.3,  0.5),   # xfmr ← bus 10
        33: ( 0.2, -1.5),   # xfmr ← bus 19
        34: (-0.5, -2.5),   # xfmr ← bus 20
        35: ( 1.8, -2.0),   # xfmr ← bus 22
        36: ( 2.2, -3.0),   # xfmr ← bus 23
        37: (-1.8,  2.5),   # xfmr ← bus 25
        38: (-2.0, -1.8),   # xfmr ← bus 29
    },
    "IEEE 69-Bus (Das et al.)": {
        # Main trunk (buses 1–27) at y = 1.6, evenly spaced x from -2 to +2
        **{n: (-2.0 + (n - 1) * 4.0 / 26, 1.6) for n in range(1, 28)},
        # Long lateral from bus 3: column A (28–47) at x = x_3, going down
        **{n: (-1.69, round(1.0 - (n - 28) * 0.1, 2)) for n in range(28, 48)},
        # Long lateral folded:    column B (48–65) at x = -2.6, going up
        **{n: (-2.6,  round(-0.9 + (n - 48) * 0.1, 2)) for n in range(48, 66)},
        # Short lateral from bus 11: buses 66–69 drop below the trunk
        66: (-0.46, 0.9), 67: (-0.46, 0.6), 68: (-0.46, 0.3), 69: (-0.46, 0.0),
    },
}

coords = CASE_COORDS.get(case_name)
if coords:
    xp = np.array([coords[b.num][0] for b in buses], dtype=float)
    yp = np.array([coords[b.num][1] for b in buses], dtype=float)
    # In a fixed layout: bus bars are horizontal, generators go up, loads go down
    _rx = np.zeros(nb)
    _ry = np.ones(nb)
    _tx = np.ones(nb)
    _ty = np.zeros(nb)
    predefined = True
else:
    ang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, nb, endpoint=False)
    xp  = np.cos(ang)
    yp  = np.sin(ang)
    _rx = np.cos(ang)
    _ry = np.sin(ang)
    _tx = -np.sin(ang)
    _ty = np.cos(ang)
    predefined = False

# ── Plotly network diagram ─────────────────────────────────────────────────────
fig = go.Figure()

# ── Branch thermal loading ─────────────────────────────────────────────────────
S_mva = [float(np.hypot(P_from[j], Q_from[j])) for j in range(len(branches))]

def _br_color(br: "Branch", smva: float) -> str:
    """Return hex color for a branch based on thermal loading fraction."""
    if br.rate <= 0:
        return "#444444" if (br.tap == 1.0 and br.shift == 0.0) else "#6D4C41"
    pct = smva / br.rate * 100
    if pct < 75:   return "#1B5E20"   # green — normal
    if pct < 90:   return "#F57F17"   # amber — approaching limit
    if pct < 100:  return "#E65100"   # orange — near limit
    return "#B71C1C"                   # red — overloaded

# Group line segments by (color, is_transformer)
line_segs: dict[str, tuple[list, list]] = {}
xfm_segs:  dict[str, tuple[list, list]] = {}
for j, br in enumerate(branches):
    ii, kk = br.fbus - 1, br.tbus - 1
    clr = _br_color(br, S_mva[j])
    is_xfm = (br.tap != 1.0 or br.shift != 0.0)
    bucket = xfm_segs if is_xfm else line_segs
    if clr not in bucket:
        bucket[clr] = ([], [])
    bucket[clr][0].extend([xp[ii], xp[kk], None])
    bucket[clr][1].extend([yp[ii], yp[kk], None])

for clr, (xl, yl) in line_segs.items():
    fig.add_trace(go.Scatter(
        x=xl, y=yl, mode="lines",
        line=dict(color=clr, width=2.0),
        hoverinfo="skip", showlegend=False,
    ))

_xfm_legend_shown = False
for clr, (xl, yl) in xfm_segs.items():
    fig.add_trace(go.Scatter(
        x=xl, y=yl, mode="lines",
        line=dict(color=clr, width=2.5, dash="dash"),
        hoverinfo="skip",
        name="Transformer" if not _xfm_legend_shown else "",
        showlegend=not _xfm_legend_shown,
    ))
    _xfm_legend_shown = True

# Branch midpoint hover hit-boxes
brhov_x, brhov_y, brhov_tip = [], [], []
for j, br in enumerate(branches):
    ii, kk = br.fbus - 1, br.tbus - 1
    mx = float((xp[ii] + xp[kk]) / 2)
    my = float((yp[ii] + yp[kk]) / 2)
    s = S_mva[j]
    is_xfm = (br.tap != 1.0 or br.shift != 0.0)
    btype = "Transformer" if is_xfm else "Line"
    if br.rate > 0:
        pct = s / br.rate * 100
        alarm = " ⚠" if pct >= 90 else ""
        loading_str = f"{pct:.1f}% of {br.rate:.0f} MVA"
    else:
        alarm = ""
        loading_str = "no rating set"
    tip = (
        f"<b>{btype} {br.fbus}→{br.tbus}{alarm}</b>"
        f"<br>P = {P_from[j]:.1f} MW    Q = {Q_from[j]:.1f} MVAr"
        f"<br>|S| = {s:.1f} MVA    Loading: {loading_str}"
    )
    if br.tap != 1.0:
        tip += f"<br>Tap = {br.tap:.3f}"
    brhov_x.append(mx)
    brhov_y.append(my)
    brhov_tip.append(tip + "<extra></extra>")

fig.add_trace(go.Scatter(
    x=brhov_x, y=brhov_y, mode="markers",
    marker=dict(size=14, color="rgba(0,0,0,0)"),
    hovertemplate=brhov_tip,
    showlegend=False,
))

# Transformer winding symbol: two touching circles at each transformer midpoint
for br in branches:
    if br.tap != 1.0 or br.shift != 0.0:
        ii, kk = br.fbus - 1, br.tbus - 1
        mx_t = (xp[ii] + xp[kk]) / 2
        my_t = (yp[ii] + yp[kk]) / 2
        L_t = np.hypot(xp[kk] - xp[ii], yp[kk] - yp[ii]) + 1e-9
        ux_t = (xp[kk] - xp[ii]) / L_t
        uy_t = (yp[kk] - yp[ii]) / L_t
        cr = 0.045
        for sign in (-1, 1):
            cx = mx_t + sign * cr * ux_t
            cy = my_t + sign * cr * uy_t
            fig.add_shape(type="circle",
                x0=cx - cr, y0=cy - cr, x1=cx + cr, y1=cy + cr,
                fillcolor="white", line=dict(color="#6D4C41", width=1.8))


# ── IEEE one-line diagram: bus bars, generator circles, load triangles ─────────
large       = nb > 40
BAR_LEN     = 0.055 if large else 0.11   # half-length of bus bar
GEN_OFF     = 0.30  if large else 0.40   # bus-center → generator circle center
GEN_R       = 0.045 if large else 0.075  # generator circle radius
LOAD_OFF    = 0.22  if large else 0.34   # bus-center → load triangle center
SHOW_SYM    = not large                  # full symbols only for smaller systems

# Colours by bus type
BAR_CLR = {BusType.SLACK: "#8B6914", BusType.PV: "#1B5E20", BusType.PQ: "#1A237E"}
BAR_W   = {BusType.SLACK: 7,         BusType.PV: 6,         BusType.PQ: 4}

# Collect load-triangle scatter points
load_xs, load_ys, load_tips = [], [], []
# Collect hover-hit-boxes
hover_xs, hover_ys, hover_tips = [], [], []

for ii, bus in enumerate(buses):
    rx, ry = _rx[ii], _ry[ii]
    tx, ty = _tx[ii], _ty[ii]

    viol    = violations[ii]
    bclr    = "#B71C1C" if viol else BAR_CLR[bus.type]
    bw      = BAR_W[bus.type]

    # ── Bus bar (tangential line segment) ──────────────────────────────────────
    fig.add_shape(type="line",
        x0=xp[ii] - BAR_LEN * tx, y0=yp[ii] - BAR_LEN * ty,
        x1=xp[ii] + BAR_LEN * tx, y1=yp[ii] + BAR_LEN * ty,
        line=dict(color=bclr, width=bw))

    # ── Bus number label ───────────────────────────────────────────────────────
    if predefined:
        # Above the bus bar centre
        num_x, num_y = xp[ii], yp[ii] + (0.10 if large else 0.16)
    else:
        # Tip of the bus bar (tangential) + slight radial push
        num_x = xp[ii] + (BAR_LEN + 0.03) * tx + 0.04 * rx
        num_y = yp[ii] + (BAR_LEN + 0.03) * ty + 0.04 * ry
    fig.add_annotation(
        x=num_x, y=num_y,
        text=f"<b>{bus.num}</b>",
        showarrow=False,
        font=dict(size=max(1, round((7 if large else 10) * label_size)), color=bclr),
    )

    # ── Voltage & angle label (skip on large predefined systems) ──────────────
    if not (large and predefined):
        v_off = 0.05 if large else 0.09
        viol_flag = " ⚠" if viol else ""
        fig.add_annotation(
            x=xp[ii] - v_off * rx,
            y=yp[ii] - v_off * ry,
            text=f"{Vm[ii]:.3f}∠{Va[ii]:.1f}°{viol_flag}",
            showarrow=False,
            font=dict(size=max(1, round((6 if large else 7.5) * label_size)),
                      color="#C62828" if viol else "#424242"),
            bgcolor="rgba(255,255,255,0.85)", borderpad=1,
        )

    # ── Generator symbol (outward side) ───────────────────────────────────────
    is_gen = bus.type in (BusType.SLACK, BusType.PV)
    if is_gen and SHOW_SYM:
        gx = xp[ii] + GEN_OFF * rx
        gy = yp[ii] + GEN_OFF * ry
        g_fill = "#FFF9C4" if bus.type == BusType.SLACK else "#E8F5E9"

        # Stub line: bus bar → generator circle edge
        fig.add_shape(type="line",
            x0=xp[ii], y0=yp[ii],
            x1=gx - GEN_R * rx, y1=gy - GEN_R * ry,
            line=dict(color=bclr, width=1.5))

        # Generator circle
        fig.add_shape(type="circle",
            x0=gx - GEN_R, y0=gy - GEN_R,
            x1=gx + GEN_R, y1=gy + GEN_R,
            fillcolor=g_fill, line=dict(color=bclr, width=2.0))

        # Sine-wave "~" label centred in circle
        fig.add_annotation(x=gx, y=gy, text="~",
            showarrow=False, font=dict(size=max(1, round(11 * label_size)), color=bclr))

        # Power output label beyond the generator circle
        lbl_off = GEN_OFF + GEN_R + 0.14
        pg_txt = (f"<b>G{bus.num}</b><br>{Pg_solved[ii]:.0f} MW<br>"
                  f"{Qg_solved[ii]:.0f} MVAr")
        g_bg  = "rgba(255,249,196,0.92)" if bus.type == BusType.SLACK else "rgba(232,245,233,0.92)"
        g_bc  = "#FFD54F" if bus.type == BusType.SLACK else "#A5D6A7"
        fig.add_annotation(
            x=xp[ii] + lbl_off * rx, y=yp[ii] + lbl_off * ry,
            text=pg_txt, showarrow=False,
            font=dict(size=max(1, round(8 * label_size)), color=bclr),
            bgcolor=g_bg, bordercolor=g_bc, borderwidth=1, borderpad=2,
            align="center",
        )
    elif is_gen and large:
        # Large system: just a compact label outside
        lbl_off = 0.22
        fig.add_annotation(
            x=xp[ii] + lbl_off * rx, y=yp[ii] + lbl_off * ry,
            text=f"{Pg_solved[ii]:.1f} MW",
            showarrow=False, font=dict(size=max(1, round(6 * label_size)), color=bclr),
            bgcolor="rgba(255,255,255,0.8)", borderpad=1,
        )

    # ── Load symbol (inward side) ──────────────────────────────────────────────
    if bus.Pd > 0 and SHOW_SYM:
        lx = xp[ii] - LOAD_OFF * rx
        ly = yp[ii] - LOAD_OFF * ry

        # Stub line: bus bar → load triangle
        fig.add_shape(type="line",
            x0=xp[ii], y0=yp[ii], x1=lx, y1=ly,
            line=dict(color="#B71C1C", width=1.5))

        # Collect load triangle points (batch into one scatter trace later)
        load_xs.append(lx)
        load_ys.append(ly)
        load_tips.append(
            f"<b>Load — Bus {bus.num}</b><br>"
            f"Pd = {bus.Pd:.2f} MW<br>Qd = {bus.Qd:.2f} MVAr"
            "<extra></extra>"
        )

        # Load label beyond the triangle
        lbl_off = LOAD_OFF + 0.12
        fig.add_annotation(
            x=xp[ii] - lbl_off * rx, y=yp[ii] - lbl_off * ry,
            text=f"↓{bus.Pd:.1f} MW<br>{bus.Qd:.1f} MVAr",
            showarrow=False, font=dict(size=max(1, round(8 * label_size)), color="#B71C1C"),
            bgcolor="rgba(255,235,238,0.9)", bordercolor="#EF9A9A",
            borderwidth=1, borderpad=2, align="center",
        )

    # ── Hover hit-box (invisible point so bus bar is hoverable) ────────────────
    tip = (
        f"<b>Bus {bus.num} — {bus.type.name}</b>"
        + (" ⚠️ VOLTAGE VIOLATION" if viol else "") +
        f"<br>V = {Vm[ii]:.4f} pu    δ = {Va[ii]:.2f}°"
    )
    if is_gen:
        tip += f"<br>Pg = {Pg_solved[ii]:.1f} MW    Qg = {Qg_solved[ii]:.1f} MVAr"
    if bus.Pd > 0:
        tip += f"<br>Pd = {bus.Pd:.2f} MW    Qd = {bus.Qd:.2f} MVAr"
    hover_xs.append(xp[ii])
    hover_ys.append(yp[ii])
    hover_tips.append(tip + "<extra></extra>")

# ── Batch load triangles ────────────────────────────────────────────────────────
if load_xs:
    fig.add_trace(go.Scatter(
        x=load_xs, y=load_ys, mode="markers",
        marker=dict(symbol="triangle-down", size=16,
                    color="#FFCDD2", line=dict(color="#B71C1C", width=2)),
        hovertemplate=load_tips, name="Load",
        showlegend=True, legendgroup="load",
    ))

# ── Invisible hover points for bus bars ────────────────────────────────────────
fig.add_trace(go.Scatter(
    x=hover_xs, y=hover_ys, mode="markers",
    marker=dict(size=14, color="rgba(0,0,0,0)"),
    hovertemplate=hover_tips, showlegend=False,
))

# ── Legend-only dummy traces for bus bar colours ───────────────────────────────
for lname, lclr in [
    ("Slack bus (reference)",    "#8B6914"),
    ("PV bus (generator)",       "#1B5E20"),
    ("PQ bus",                   "#1A237E"),
]:
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="line-ew", size=16, color=lclr,
                    line=dict(color=lclr, width=5)),
        name=lname, showlegend=True,
    ))
if SHOW_SYM:
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="circle-open", size=14, color="#1B5E20",
                    line=dict(color="#1B5E20", width=2)),
        name="Generator (circle ~)", showlegend=True,
    ))

# Thermal loading legend (only if any branch has a rating)
if any(br.rate > 0 for br in branches):
    for lname, lclr in [
        ("< 75% loaded",   "#1B5E20"),
        ("75–90% loaded",  "#F57F17"),
        ("> 90% loaded",   "#E65100"),
        ("Overloaded",     "#B71C1C"),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=lclr, width=4),
            name=lname, showlegend=True,
        ))

if predefined:
    # Pad to fit generator labels above and load labels below
    sym_pad = 0.75 if SHOW_SYM else 0.35
    x_lo, x_hi = xp.min() - 0.4, xp.max() + 0.4
    y_lo, y_hi = yp.min() - sym_pad, yp.max() + sym_pad
    # Enforce equal span so topology isn't stretched
    span = max(x_hi - x_lo, y_hi - y_lo)
    xc, yc = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
    xr = [xc - span / 2, xc + span / 2]
    yr = [yc - span / 2, yc + span / 2]
else:
    pad = 2.2
    xr = yr = [-pad, pad]

fig.update_layout(
    showlegend=True,
    legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.92)",
                bordercolor="#cccccc", borderwidth=1, font=dict(size=10)),
    xaxis=dict(visible=False, range=xr),
    yaxis=dict(visible=False, range=yr, scaleanchor="x", scaleratio=1),
    plot_bgcolor="white", paper_bgcolor="white",
    height=640,
    margin=dict(l=10, r=10, t=48, b=10),
    title=dict(
        text="Network diagram — branch color shows thermal loading  |  hover for details",
        font=dict(size=13), x=0.5, xanchor="center",
    ),
)

# ── Tab layout ─────────────────────────────────────────────────────────────────
tab_net, tab_vp, tab_n1, tab_opf, tab_stab = st.tabs(
    ["🔌 Network Diagram", "📊 Voltage Profile", "🔄 N-1 Security", "⚡ OPF", "🌊 Stability"]
)

with tab_net:
    st.plotly_chart(fig, use_container_width=True,
                    config={"scrollZoom": True, "displayModeBar": True})

    # Key metrics
    total_gen  = sum(Pg_solved[i] for i, b in enumerate(buses)
                     if b.type in (BusType.SLACK, BusType.PV))
    total_load = sum(b.Pd for b in buses)
    total_loss = sum(abs(p) for p in P_loss)
    loss_pct   = 100 * total_loss / total_load if total_load > 0 else 0.0
    min_v      = float(Vm.min())
    max_v      = float(Vm.max())
    n_viol     = int(violations.sum())
    max_load_pct = max(
        (S_mva[j] / branches[j].rate * 100 for j in range(len(branches))
         if branches[j].rate > 0),
        default=float("nan"),
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total generation", f"{total_gen:.1f} MW")
    c2.metric("Total load",       f"{total_load:.1f} MW")
    c3.metric("System losses",    f"{total_loss:.2f} MW",
              f"{loss_pct:.2f}% of load")
    c4.metric("Min voltage",      f"{min_v:.4f} pu",
              delta="⚠ below 0.95" if min_v < V_LO else "within limits",
              delta_color="inverse" if min_v < V_LO else "off")
    c5.metric("Max voltage",      f"{max_v:.4f} pu",
              delta="⚠ above 1.05" if max_v > V_HI else "within limits",
              delta_color="inverse" if max_v > V_HI else "off")
    c6.metric("Max branch load",
              f"{max_load_pct:.1f}%" if not np.isnan(max_load_pct) else "—",
              delta="⚠ overloaded" if max_load_pct >= 100 else (
                  "⚠ near limit" if max_load_pct >= 90 else "within limits"),
              delta_color="inverse" if max_load_pct >= 90 else "off")

    st.divider()
    col_bus, col_br = st.columns(2)

    with col_bus:
        st.subheader("Bus summary")
        bus_rows = []
        for ii, b in enumerate(buses):
            viol_str = " ⚠" if violations[ii] else ""
            bus_rows.append({
                "Bus":       b.num,
                "Type":      b.type.name,
                "V (pu)":    f"{Vm[ii]:.4f}{viol_str}",
                "δ (°)":     f"{Va[ii]:.2f}",
                "Pg (MW)":   f"{Pg_solved[ii]:.1f}"  if b.type in (BusType.SLACK, BusType.PV) else "—",
                "Qg (MVAr)": f"{Qg_solved[ii]:.1f}"  if b.type in (BusType.SLACK, BusType.PV) else "—",
                "Pd (MW)":   f"{b.Pd:.2f}"            if b.Pd  else "—",
                "Qd (MVAr)": f"{b.Qd:.2f}"            if b.Qd  else "—",
            })
        st.dataframe(pd.DataFrame(bus_rows), hide_index=True,
                     use_container_width=True)

    with col_br:
        st.subheader("Branch flows")
        br_rows = []
        for j, br in enumerate(branches):
            s = S_mva[j]
            if br.rate > 0:
                pct = s / br.rate * 100
                load_str = f"{pct:.1f}%"
                alarm = " ⚠" if pct >= 90 else ""
            else:
                load_str = "—"
                alarm = ""
            is_xfm = br.tap != 1.0 or br.shift != 0.0
            br_rows.append({
                "Branch":          f"{br.fbus}→{br.tbus}" + (" (T)" if is_xfm else ""),
                "P (MW)":          f"{P_from[j]:.1f}",
                "Q (MVAr)":        f"{Q_from[j]:.1f}",
                "|S| (MVA)":       f"{s:.1f}",
                "Rating (MVA)":    f"{br.rate:.0f}" if br.rate > 0 else "—",
                "Loading":         load_str + alarm,
                "Loss (MW)":       f"{abs(P_loss[j]):.2f}",
            })
        st.dataframe(pd.DataFrame(br_rows), hide_index=True,
                     use_container_width=True)

    with st.expander("What am I looking at?", expanded=False):
        st.markdown("""
**Bus types**
| Color | Type | What it means |
|---|---|---|
| Gold bar | **Slack** | Reference bus — absorbs whatever power the network needs to balance. Voltage magnitude and angle are fixed. |
| Green bar | **PV** | Generator bus — real power (Pg) and voltage magnitude are specified. |
| Blue bar | **PQ** | Load bus — real and reactive power consumption are specified. |

**Branch thermal coloring**
| Color | Meaning |
|---|---|
| Dark green | < 75% of thermal rating — normal |
| Amber | 75–90% — approaching limit |
| Orange | 90–100% — near limit |
| Red | > 100% — overloaded |
| Gray / brown (dashed) | No rating set |

Hover over any bus bar or branch midpoint for live details.
        """)

# ── Voltage Profile tab ────────────────────────────────────────────────────────
with tab_vp:
    st.subheader("Bus voltage profile")

    sort_by = st.radio("Sort by", ["Bus number", "Voltage (low→high)"],
                       horizontal=True)
    bus_nums = [b.num for b in buses]
    vms = list(Vm)

    if sort_by == "Voltage (low→high)":
        order = sorted(range(nb), key=lambda i: vms[i])
    else:
        order = list(range(nb))

    x_labels = [str(buses[i].num) for i in order]
    y_vals   = [vms[i] for i in order]
    bar_clrs = []
    for v in y_vals:
        if v < 0.90:   bar_clrs.append("#B71C1C")    # severe under
        elif v < V_LO: bar_clrs.append("#E65100")    # under
        elif v > 1.07: bar_clrs.append("#B71C1C")    # severe over
        elif v > V_HI: bar_clrs.append("#E65100")    # over
        elif v < 0.97: bar_clrs.append("#F57F17")    # approaching low
        elif v > 1.03: bar_clrs.append("#F57F17")    # approaching high
        else:          bar_clrs.append("#1B5E20")    # normal

    vfig = go.Figure()
    vfig.add_trace(go.Bar(
        x=x_labels, y=y_vals,
        marker_color=bar_clrs,
        hovertemplate="Bus %{x}<br>V = %{y:.4f} pu<extra></extra>",
        name="Voltage",
    ))
    vfig.add_hline(y=V_LO, line_color="#E65100", line_dash="dash",
                   annotation_text="0.95 pu low limit",
                   annotation_position="top right")
    vfig.add_hline(y=V_HI, line_color="#E65100", line_dash="dash",
                   annotation_text="1.05 pu high limit",
                   annotation_position="top right")
    vfig.add_hline(y=1.0, line_color="#888888", line_dash="dot",
                   annotation_text="1.0 pu nominal")
    vfig.update_layout(
        xaxis_title="Bus number",
        yaxis_title="Voltage magnitude (pu)",
        yaxis=dict(range=[min(0.88, float(Vm.min()) - 0.02),
                          max(1.12, float(Vm.max()) + 0.02)]),
        height=420,
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=40),
    )
    st.plotly_chart(vfig, use_container_width=True)

    n_viol_lo = int((Vm < V_LO).sum())
    n_viol_hi = int((Vm > V_HI).sum())
    if n_viol_lo + n_viol_hi == 0:
        st.success(f"All {nb} buses within 0.95–1.05 pu.")
    else:
        if n_viol_lo:
            st.error(f"{n_viol_lo} bus(es) below 0.95 pu: "
                     + ", ".join(str(buses[i].num)
                                 for i in range(nb) if Vm[i] < V_LO))
        if n_viol_hi:
            st.error(f"{n_viol_hi} bus(es) above 1.05 pu: "
                     + ", ".join(str(buses[i].num)
                                 for i in range(nb) if Vm[i] > V_HI))

# ── N-1 Security tab ──────────────────────────────────────────────────────────
with tab_n1:
    st.subheader("N-1 contingency analysis")

    is_radial = any(k in case_name for k in ("33", "69"))

    if is_radial:
        st.warning(
            "**This feeder is radial.** Removing any single branch isolates a "
            "section — there is no N-1 security without open-tie switching or "
            "network reconfiguration. This is the defining challenge of "
            "distribution protection design."
        )
        st.info(
            "N-1 analysis is enabled for meshed transmission systems "
            "(IEEE 9-Bus, 14-Bus, 39-Bus). Switch to one of those to explore "
            "contingency security."
        )
    else:
        st.markdown(
            "Remove each branch one at a time and re-solve. "
            "Flags any voltage violations (< 0.95 or > 1.05 pu) or "
            "thermal overloads (> 90% of rating) in the surviving network."
        )
        if st.button("▶ Run N-1 Analysis", type="primary"):
            n1_rows = []
            prog = st.progress(0.0, text="Running contingencies…")
            import copy as _copy

            for j, br_out in enumerate(branches):
                prog.progress((j + 1) / len(branches),
                              text=f"Contingency {j+1}/{len(branches)}: "
                                   f"branch {br_out.fbus}→{br_out.tbus}")

                buses_c    = _copy.deepcopy(buses)
                gens_c     = _copy.deepcopy(gens)
                branches_c = [b for k, b in enumerate(branches) if k != j]

                try:
                    res_c = newton_raphson_pf(
                        buses_c, gens_c, branches_c,
                        baseMVA=100.0, tol=1e-6, max_it=60, verbose=False,
                    )
                    if not res_c["converged"]:
                        status = "No solution"
                        min_v_c = float("nan")
                        max_v_c = float("nan")
                        max_ld  = float("nan")
                    else:
                        Vm_c    = res_c["Vm"]
                        Va_c    = res_c["Va"]
                        nb_c    = len(buses_c)
                        Y_c     = build_ybus(nb_c, branches_c,
                                             {b.num: b for b in buses_c})
                        fl_c    = compute_branch_flows(Y_c, Vm_c, Va_c,
                                                       branches_c, 100.0)
                        Pf_c    = [float(fl_c["P_from (MW)"].iloc[k]) for k in range(len(branches_c))]
                        Qf_c    = [float(fl_c["Q_from (MVAr)"].iloc[k]) for k in range(len(branches_c))]
                        min_v_c = float(Vm_c.min())
                        max_v_c = float(Vm_c.max())
                        v_ok    = (min_v_c >= V_LO) and (max_v_c <= V_HI)
                        max_ld  = max(
                            (np.hypot(Pf_c[k], Qf_c[k]) / branches_c[k].rate * 100
                             for k in range(len(branches_c))
                             if branches_c[k].rate > 0),
                            default=float("nan"),
                        )
                        status = "OK"
                        if not v_ok: status = "Voltage violation"
                        if not np.isnan(max_ld) and max_ld >= 100:
                            status = "Thermal overload" if status == "OK" \
                                     else status + " + Thermal overload"
                except Exception:
                    status = "Solver error"
                    min_v_c = max_v_c = max_ld = float("nan")

                is_xfm = br_out.tap != 1.0 or br_out.shift != 0.0
                n1_rows.append({
                    "Outage":         f"{br_out.fbus}→{br_out.tbus}" + (" (T)" if is_xfm else ""),
                    "Status":         status,
                    "Min V (pu)":     f"{min_v_c:.4f}" if not np.isnan(min_v_c) else "—",
                    "Max V (pu)":     f"{max_v_c:.4f}" if not np.isnan(max_v_c) else "—",
                    "Max loading (%)": f"{max_ld:.1f}" if not np.isnan(max_ld) else "—",
                })

            prog.empty()
            df_n1 = pd.DataFrame(n1_rows)
            st.success(f"Completed {len(branches)} contingencies.")

            # Summary
            ok_count   = (df_n1["Status"] == "OK").sum()
            bad_count  = len(df_n1) - ok_count
            cc1, cc2 = st.columns(2)
            cc1.metric("Secure contingencies",   f"{ok_count} / {len(branches)}")
            cc2.metric("Contingencies with issues", str(bad_count),
                       delta_color="inverse" if bad_count > 0 else "off")

            # Style the dataframe
            def _status_color(val):
                if val == "OK": return "color: #1B5E20"
                if val == "No solution": return "color: #B71C1C; font-weight: bold"
                return "color: #E65100; font-weight: bold"

            st.dataframe(
                df_n1.style.applymap(_status_color, subset=["Status"]),
                hide_index=True, use_container_width=True,
            )
        else:
            st.info("Click ▶ Run N-1 Analysis to evaluate each single-branch outage.")

# ── OPF tab ────────────────────────────────────────────────────────────────────
with tab_opf:
    st.subheader("DC Optimal Power Flow & Locational Marginal Prices")

    opf_gens = case_opf_gens(case_name)

    if opf_gens is None:
        st.warning(
            "OPF applies to meshed transmission systems. "
            "Switch to IEEE 9-Bus, 14-Bus, or 39-Bus to explore LMPs and congestion."
        )
    else:
        total_load_opf = sum(b.Pd for b in buses)
        total_pmax_opf = sum(og.Pmax for og in opf_gens)
        total_pmin_opf = sum(og.Pmin for og in opf_gens)

        if total_load_opf > total_pmax_opf:
            st.error(
                f"Total load ({total_load_opf:.1f} MW) exceeds generator capacity "
                f"({total_pmax_opf:.1f} MW). Reduce the load scale."
            )
        elif total_load_opf < total_pmin_opf:
            st.error(
                f"Total load ({total_load_opf:.1f} MW) is below committed minimum "
                f"({total_pmin_opf:.1f} MW)."
            )
        else:
            res_opf = dc_opf(buses, branches, opf_gens, enforce_limits=True)
            res_unc = dc_opf(buses, branches, opf_gens, enforce_limits=False)

            if not res_opf["converged"]:
                st.error(f"OPF did not converge: {res_opf['message']}")
                st.info("Try reducing the load scale or increasing line ratings.")
            else:
                LMP    = res_opf["LMP"]
                Pg_opf = res_opf["Pg_mw"]
                Pg_unc = res_unc["Pg_mw"] if res_unc["converged"] else np.full(len(opf_gens), float("nan"))
                Pl_opf = res_opf["P_line_mw"]

                lmp_min  = float(LMP.min())
                lmp_max  = float(LMP.max())
                lmp_spread = lmp_max - lmp_min
                congested_idx = [j for j, br in enumerate(branches)
                                 if br.rate > 0 and abs(Pl_opf[j]) >= 0.999 * br.rate]
                cost_gap = (res_opf["total_cost"] - res_unc["total_cost"]
                            if res_unc["converged"] else float("nan"))

                # ── Metrics ───────────────────────────────────────────────────
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("OPF total cost", f"${res_opf['total_cost']:,.0f}/hr")
                mc2.metric("Unconstrained cost",
                           f"${res_unc['total_cost']:,.0f}/hr" if res_unc["converged"] else "—")
                mc3.metric("Congestion cost",
                           f"${cost_gap:,.0f}/hr" if not np.isnan(cost_gap) else "—",
                           delta="no congestion" if (not np.isnan(cost_gap) and cost_gap < 1) else None,
                           delta_color="off")
                mc4.metric("LMP spread",
                           f"${lmp_min:.2f}–${lmp_max:.2f}/MWh",
                           delta=f"{len(congested_idx)} congested branch(es)" if congested_idx else "no congestion",
                           delta_color="inverse" if congested_idx else "off")

                if len(congested_idx) == 0:
                    cheapest = min(opf_gens, key=lambda g: g.c1)
                    st.info(
                        f"**No congestion** — transmission limits aren't binding, so all buses see "
                        f"the same marginal cost. The system price is set by **{cheapest.name}** "
                        f"at ${cheapest.c1:.0f}/MWh. Every bus would benefit equally from an additional "
                        f"MW of generation anywhere on the system."
                    )
                else:
                    high_i = int(np.argmax(LMP))
                    low_i  = int(np.argmin(LMP))
                    st.warning(
                        f"**Congestion on {len(congested_idx)} branch(es)** — "
                        f"Bus {buses[low_i].num} is the low-cost location (${lmp_min:.2f}/MWh) and "
                        f"Bus {buses[high_i].num} is the high-cost location (${lmp_max:.2f}/MWh). "
                        f"Transmission limits are preventing cheap generation from reaching all loads, "
                        f"forcing more expensive units to pick up the slack locally. "
                        f"The ${cost_gap:,.0f}/hr congestion cost is the premium the system pays for that inefficiency."
                    )

                st.divider()

                # ── Generator dispatch comparison ─────────────────────────────
                col_gen, col_lmp = st.columns([3, 2])

                with col_gen:
                    st.markdown("**Generator dispatch — unconstrained vs OPF**")
                    gen_rows = []
                    for k, og in enumerate(opf_gens):
                        pu = Pg_unc[k]
                        po = Pg_opf[k]
                        delta = float(po - pu) if not np.isnan(pu) else float("nan")
                        gen_rows.append({
                            "Generator":           og.name or f"G{og.bus}",
                            "Cost ($/MWh)":        f"{og.c1:.0f}",
                            "Pmin–Pmax (MW)":      f"{og.Pmin:.0f}–{og.Pmax:.0f}",
                            "Unconstrained (MW)":  f"{pu:.1f}" if not np.isnan(pu) else "—",
                            "OPF (MW)":            f"{po:.1f}",
                            "Δ (MW)":              f"{delta:+.1f}" if not np.isnan(delta) else "—",
                        })
                    st.dataframe(pd.DataFrame(gen_rows), hide_index=True,
                                 use_container_width=True)

                with col_lmp:
                    st.markdown("**LMP by bus (sorted high→low)**")
                    lmp_rows = [{"Bus": b.num, "LMP ($/MWh)": f"{LMP[i]:.2f}",
                                 "Type": b.type.name}
                                for i, b in enumerate(buses)]
                    lmp_df = (pd.DataFrame(lmp_rows)
                              .sort_values("LMP ($/MWh)", ascending=False)
                              .reset_index(drop=True))
                    st.dataframe(lmp_df, hide_index=True, use_container_width=True)

                st.divider()

                # ── LMP bar chart ─────────────────────────────────────────────
                st.markdown("**LMP by bus**")
                lmp_norm = (LMP - lmp_min) / (lmp_spread + 1e-9)
                # Interpolate red→yellow→green (reversed: high LMP = red)
                def _lmp_color(v):
                    r = int(min(255, 2 * v * 255))
                    g = int(min(255, 2 * (1 - v) * 255))
                    return f"rgb({r},{g},50)"

                lmp_bar_colors = [_lmp_color(float(lmp_norm[i])) for i in range(nb)]
                lmp_fig = go.Figure(go.Bar(
                    x=[str(b.num) for b in buses],
                    y=[float(LMP[i]) for i in range(nb)],
                    marker_color=lmp_bar_colors,
                    hovertemplate="Bus %{x}<br>LMP = $%{y:.2f}/MWh<extra></extra>",
                ))
                lmp_fig.add_hline(
                    y=float(LMP[next(i for i, b in enumerate(buses) if b.type.name == "SLACK")]),
                    line_dash="dot", line_color="#666",
                    annotation_text="slack bus LMP",
                    annotation_position="top right",
                )
                lmp_fig.update_layout(
                    xaxis_title="Bus", yaxis_title="LMP ($/MWh)",
                    height=280, plot_bgcolor="white", paper_bgcolor="white",
                    showlegend=False, margin=dict(l=10, r=10, t=10, b=40),
                )
                st.plotly_chart(lmp_fig, use_container_width=True)

                # ── LMP network map ───────────────────────────────────────────
                st.markdown("**LMP network map** — node color = LMP, line color = thermal loading")
                lmp_net = go.Figure()

                for j, br in enumerate(branches):
                    ii, kk = br.fbus - 1, br.tbus - 1
                    is_xfm = (br.tap != 1.0 or br.shift != 0.0)
                    if br.rate > 0:
                        pct = abs(Pl_opf[j]) / br.rate * 100
                        if   pct >= 100: lclr = "#B71C1C"
                        elif pct >=  90: lclr = "#E65100"
                        elif pct >=  75: lclr = "#F57F17"
                        else:            lclr = "#1B5E20"
                        lw = 3.5 if j in congested_idx else 1.8
                    else:
                        lclr, lw = "#AAAAAA", 1.5
                    lmp_net.add_trace(go.Scatter(
                        x=[xp[br.fbus - 1], xp[br.tbus - 1], None],
                        y=[yp[br.fbus - 1], yp[br.tbus - 1], None],
                        mode="lines",
                        line=dict(color=lclr, width=lw,
                                  dash="dash" if is_xfm else "solid"),
                        hoverinfo="skip", showlegend=False,
                    ))

                lmp_arr = np.array([float(LMP[i]) for i in range(nb)])
                bus_hover = [
                    f"<b>Bus {b.num}</b><br>LMP = ${float(LMP[i]):.2f}/MWh"
                    + (f"<br>Pd = {b.Pd:.1f} MW" if b.Pd > 0 else "")
                    + "<extra></extra>"
                    for i, b in enumerate(buses)
                ]
                lmp_net.add_trace(go.Scatter(
                    x=list(xp), y=list(yp), mode="markers",
                    marker=dict(
                        size=16,
                        color=lmp_arr,
                        colorscale="RdYlGn_r",
                        cmin=lmp_min, cmax=lmp_max,
                        showscale=True,
                        colorbar=dict(title="LMP<br>($/MWh)", thickness=14,
                                      len=0.75, x=1.01),
                        line=dict(color="white", width=1.5),
                    ),
                    hovertemplate=bus_hover,
                    showlegend=False,
                ))

                # LMP labels on each bus node
                for i, b in enumerate(buses):
                    lmp_net.add_annotation(
                        x=xp[i], y=yp[i],
                        text=f"${float(LMP[i]):.1f}",
                        showarrow=False,
                        yanchor="bottom",
                        yshift=11,
                        font=dict(size=max(1, round(9 * label_size)), color="#222222"),
                        bgcolor="rgba(255,255,255,0.75)",
                        borderpad=1,
                    )

                # Mark congested branches with a bold ⚡ annotation
                for j in congested_idx:
                    br = branches[j]
                    mx = (xp[br.fbus - 1] + xp[br.tbus - 1]) / 2
                    my = (yp[br.fbus - 1] + yp[br.tbus - 1]) / 2
                    lmp_net.add_annotation(x=mx, y=my, text="⚡",
                                           showarrow=False, font=dict(size=14))

                lmp_net.update_layout(
                    xaxis=dict(visible=False, range=xr),
                    yaxis=dict(visible=False, range=yr, scaleanchor="x", scaleratio=1),
                    plot_bgcolor="white", paper_bgcolor="white",
                    height=500, margin=dict(l=10, r=60, t=30, b=10),
                    title=dict(
                        text="Node color = LMP  |  line color = thermal loading  |  ⚡ = at-limit branch",
                        font=dict(size=12), x=0.5, xanchor="center",
                    ),
                )
                st.plotly_chart(lmp_net, use_container_width=True,
                                config={"scrollZoom": True, "displayModeBar": True})

                # ── Explainer ─────────────────────────────────────────────────
                with st.expander("Understanding OPF and LMPs", expanded=False):
                    st.markdown("""
**DC Optimal Power Flow** minimizes total generation cost subject to three sets of constraints:
- **Power balance** at every bus (Kirchhoff's current law in DC approximation)
- **Generator output limits** — each unit stays within its Pmin / Pmax
- **Line thermal limits** — each branch stays within its MW rating

The "DC" approximation linearizes the power flow by assuming flat voltage (1.0 pu everywhere)
and ignoring resistance. This turns the problem into a linear program (LP) that solves in
milliseconds and has clean, interpretable dual variables.

---

**Locational Marginal Price (LMP)** at bus *i* is the marginal cost of serving one additional
MWh of load at that location. It is the dual variable of the power balance constraint.

| Scenario | LMP behavior |
|---|---|
| No congestion | All bus LMPs equal the marginal generator's cost — one system price |
| Line at limit | LMPs diverge: cheap generators *behind* the line set a low LMP on their side; expensive generators on the load side set a higher LMP |
| Generator at limit | The next-cheapest unit sets the system price |

**Congestion cost** = OPF total cost − unconstrained dispatch cost. This is the premium the
system pays because cheap generation cannot reach the load.

---

**What ISOs actually do with LMPs**
- Energy market clearing: generators are paid the LMP at their injection bus
- Financial Transmission Rights (FTRs): hedges against congestion, priced at the LMP spread
- Investment signals: persistent high-LMP buses attract new generation or transmission
- FERC Order 2000 / Order 719 require ISOs to publish LMPs in real time and day-ahead

---

**Try this:** Increase the load scale slider (large cases) or reduce a line rating in the code
to create congestion. Watch the LMPs diverge and note which generators are redispatched.
                    """)

# ── Stability tab ──────────────────────────────────────────────────────────────
with tab_stab:
    st.subheader("Transient Stability — Classical Machine Model")

    stab_gens = default_stab_gens(case_name)

    if stab_gens is None:
        st.warning(
            "Transient stability applies to meshed transmission systems with generators. "
            "Switch to IEEE 9-Bus, 14-Bus, or 39-Bus."
        )
    else:
        st.caption(
            f"Classical model (constant E' behind X'd) · {len(stab_gens)} generators · "
            "H and X'd values in the expander below."
        )

        # ── Generator data table ──────────────────────────────────────────────
        with st.expander("Generator classical-model parameters", expanded=False):
            sg_rows = [{"Generator": sg.name, "Bus": sg.bus,
                        "H (s)": sg.H, "X'd (pu)": sg.Xd}
                       for sg in stab_gens]
            st.dataframe(pd.DataFrame(sg_rows), hide_index=True,
                         use_container_width=True)

        st.divider()

        # ── DER penetration settings ──────────────────────────────────────────
        st.markdown("**DER penetration scenario**")
        der_cols = st.columns([2, 2, 2])
        with der_cols[0]:
            der_pen = st.slider(
                "DER penetration (%)", 0, 80, 0, 5,
                help="Fraction of synchronous generation capacity displaced by inverter-based DERs.",
            ) / 100.0
        with der_cols[1]:
            der_type = st.radio(
                "DER model", ["None", "Legacy (pre-1547)", "IEEE 1547-2018"],
                index=0, horizontal=False,
            )
            der_type_key = {"None": "none", "Legacy (pre-1547)": "legacy",
                            "IEEE 1547-2018": "1547-2018"}[der_type]
        with der_cols[2]:
            H_virt = 3.0
            if der_type_key == "1547-2018":
                H_virt = st.slider(
                    "Virtual inertia H_v (s)", 1.0, 6.0, 3.0, 0.5,
                    help="Inertia constant synthesised by grid-forming or fast-frequency-response inverters.",
                )
            else:
                st.caption("Virtual inertia: N/A")

        # Compute effective H and show delta
        H_orig_arr = np.array([sg.H for sg in stab_gens])
        H_eff_arr  = effective_H(stab_gens, der_pen, der_type_key, H_virt)
        H_sys_orig = float(H_orig_arr.sum())
        H_sys_eff  = float(H_eff_arr.sum())
        H_delta_pct = (H_sys_eff - H_sys_orig) / H_sys_orig * 100

        if der_pen > 0:
            hi1, hi2 = st.columns(2)
            hi1.metric("Effective system H", f"{H_sys_eff:.2f} s",
                       delta=f"{H_delta_pct:+.1f}% vs base",
                       delta_color="inverse" if H_delta_pct < 0 else "normal")
            hi2.metric("Synchronous H retained",
                       f"{H_sys_eff / H_sys_orig * 100:.0f}%",
                       delta=f"{der_pen*100:.0f}% displaced by DERs")

        st.divider()

        # ── Fault / simulation settings ───────────────────────────────────────
        sc1, sc2 = st.columns(2)

        with sc1:
            all_bus_nums = [b.num for b in buses]
            gen_bus_nums = {sg.bus for sg in stab_gens}
            slack_num    = next(b.num for b in buses if b.type == BusType.SLACK)
            hv_buses     = [n for n in all_bus_nums
                            if n not in gen_bus_nums and n != slack_num]
            suggest      = hv_buses if hv_buses else all_bus_nums
            default_fb   = suggest[len(suggest) // 2]

            fault_bus = st.selectbox(
                "Fault bus (3-phase fault location)", options=all_bus_nums,
                index=all_bus_nums.index(default_fb),
                format_func=lambda n: f"Bus {n}" + (
                    " [gen]" if n in gen_bus_nums else
                    " [slack]" if n == slack_num else ""),
            )
            t_clear = st.slider(
                "Fault clearing time (s)", 0.02, 1.20, 0.15, 0.01,
                help="Typical: 0.083 s (5 cycles) to 0.33 s (20 cycles).",
            )
            t_end = st.slider("Simulation duration (s)", 2.0, 10.0, 5.0, 0.5)

        with sc2:
            remove_line = st.checkbox("Remove a branch post-fault", value=False,
                help="Simulates fault clearing by tripping a line.")
            drop_idx = None
            if remove_line:
                br_labels = [f"Branch {br.fbus}→{br.tbus}" +
                             (" (T)" if (br.tap != 1.0 or br.shift != 0.0) else "")
                             for br in branches]
                sel = st.selectbox("Branch to remove post-fault",
                                   range(len(branches)),
                                   format_func=lambda i: br_labels[i])
                drop_idx = sel

        run_btn = st.button("▶ Run Stability Simulation", type="primary")

        if run_btn:
            with st.spinner("Integrating swing equations…"):
                try:
                    sres = run_stability(
                        buses, branches, stab_gens,
                        Vm, Va, Pg_solved, Qg_solved,
                        fault_bus=fault_bus, t_clear=t_clear,
                        drop_branch_idx=drop_idx, t_end=t_end,
                        baseMVA=100.0, H_override=H_eff_arr,
                    )
                    # Store ROCOF alongside the result
                    rocof_info = compute_rocof(
                        stab_gens, buses, branches, Vm, Va,
                        Pg_solved, Qg_solved, fault_bus,
                        H_eff_arr, baseMVA=100.0,
                    )
                    sres["rocof"] = rocof_info
                    sres["H_eff"] = H_eff_arr
                    sres["der_type"] = der_type_key
                    sres["der_pen"]  = der_pen
                    st.session_state["stab_result"] = sres
                    st.session_state["stab_case"]   = case_name
                except Exception as exc:
                    st.error(f"Simulation error: {exc}")
                    st.session_state.pop("stab_result", None)

        # ── Display results ───────────────────────────────────────────────────
        sres = st.session_state.get("stab_result")
        if sres is not None and st.session_state.get("stab_case") == case_name:
            stable   = sres["stable"]
            t        = sres["t"]
            ddeg     = sres["delta_deg"]      # (ng, n_t) absolute angles [°]
            drel     = sres["delta_rel"]      # (ng, n_t) relative to G1 [°]
            ng_s     = len(stab_gens)
            tc       = sres["t_clear"]

            # Status banner
            if stable:
                st.success(f"**STABLE** — all generators remain in synchronism after fault cleared at {tc:.3f} s")
            else:
                st.error(f"**UNSTABLE** — generator(s) lose synchronism (pole slipping) after fault cleared at {tc:.3f} s")

            # Quick metrics
            rocof_info = sres.get("rocof")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Fault bus", f"Bus {sres['fault_bus']}")
            mc2.metric("Clearing time", f"{tc:.3f} s  ({tc*60:.1f} cycles)")
            max_sep = float(np.max(np.abs(drel)))
            mc3.metric("Max angle separation", f"{max_sep:.1f}°",
                       delta="unstable" if not stable else "stable",
                       delta_color="inverse" if not stable else "off")
            if rocof_info:
                rocof_sys = rocof_info["rocof_sys"]
                mc4.metric("System ROCOF", f"{rocof_sys:.2f} Hz/s",
                           delta="high — may trigger ROCOF relays" if rocof_sys > 1.0 else "within typical limits",
                           delta_color="inverse" if rocof_sys > 1.0 else "off")

            # Plain-language interpretation of this specific run
            _sep_per = np.max(np.abs(drel), axis=1)
            _sep_per[0] = 0.0   # G1 is the reference — exclude it
            _max_k = int(np.argmax(_sep_per)) if ng_s > 1 else 0
            _max_name = sres["gen_names"][_max_k]
            _h_vals = [sg.H for sg in stab_gens]

            if stable:
                st.info(
                    f"**What you're seeing:** The pink band is the fault period. After clearing "
                    f"at **{tc:.3f} s ({tc*60:.1f} cycles)**, all generators pulled back into "
                    f"synchronism — angles oscillate then converge. "
                    f"**{_max_name}** swung the furthest ({max_sep:.1f}° peak), which is typical "
                    f"for the generator with the lowest inertia constant H "
                    f"(H={_h_vals[_max_k]:.2f} s — lighter rotors accelerate faster). "
                    f"The bottom plot is the key diagnostic: lines staying below the orange 120° "
                    f"dashed line mean every machine stayed in step."
                )
            else:
                st.error(
                    f"**What you're seeing:** After clearing at **{tc:.3f} s ({tc*60:.1f} cycles)**, "
                    f"**{_max_name}** crossed 120° separation and kept accelerating — this is pole "
                    f"slipping. Once a machine goes past that point, the electromagnetic restoring "
                    f"torque reverses direction and the rotor can't return to synchronism. It must "
                    f"be tripped and re-synchronized. Try reducing the clearing time or use "
                    f"'Find CCT' below to find the exact boundary."
                )

            if rocof_info:
                _rsys = rocof_info["rocof_sys"]
                _rlabel = (
                    "well above typical ROCOF relay thresholds — underfrequency load shedding or fast-frequency-response resources would likely activate"
                    if _rsys > 2.0 else
                    "may trigger ROCOF relays at some utilities (common settings: 0.5–2 Hz/s per NERC PRC-024)"
                    if _rsys > 1.0 else
                    "within typical ride-through thresholds for most ROCOF relay settings"
                )
                st.caption(f"ROCOF context: {_rsys:.2f} Hz/s is {_rlabel}.")

            # Colour palette — one colour per generator
            GEN_COLORS = ["#1565C0", "#C62828", "#2E7D32", "#6A1B9A",
                          "#E65100", "#00695C", "#4E342E", "#1565C0",
                          "#F9A825", "#37474F"]

            # ── Plot 1: absolute rotor angles ─────────────────────────────────
            st.markdown("**Rotor angle trajectories (absolute)**")
            afig = go.Figure()

            # Fault period shading
            afig.add_vrect(x0=0, x1=tc, fillcolor="rgba(255,200,200,0.35)",
                           line_width=0, annotation_text="fault",
                           annotation_position="top left")
            afig.add_vline(x=tc, line_dash="dash", line_color="#C62828",
                           annotation_text=f"cleared {tc:.3f} s",
                           annotation_position="top right")

            for k, name in enumerate(sres["gen_names"]):
                afig.add_trace(go.Scatter(
                    x=t, y=ddeg[k],
                    mode="lines", name=name,
                    line=dict(color=GEN_COLORS[k % len(GEN_COLORS)], width=2),
                    hovertemplate=f"{name}<br>t=%{{x:.3f}} s  δ=%{{y:.1f}}°<extra></extra>",
                ))

            afig.update_layout(
                xaxis_title="Time (s)", yaxis_title="Rotor angle δ (degrees)",
                height=380, plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"),
                margin=dict(l=10, r=10, t=10, b=40),
            )
            st.plotly_chart(afig, use_container_width=True)

            # ── Plot 2: angle relative to G1 ─────────────────────────────────
            st.markdown("**Relative rotor angles (δᵢ − δ_G1) — key stability indicator**")
            rfig = go.Figure()

            rfig.add_vrect(x0=0, x1=tc, fillcolor="rgba(255,200,200,0.35)",
                           line_width=0)
            rfig.add_vline(x=tc, line_dash="dash", line_color="#C62828")
            rfig.add_hline(y=120,  line_dash="dot", line_color="#E65100",
                           annotation_text="120° threshold", annotation_position="top right")
            rfig.add_hline(y=-120, line_dash="dot", line_color="#E65100")

            for k in range(ng_s):
                if k == 0:
                    continue   # G1 relative to itself is always 0
                name = sres["gen_names"][k]
                rfig.add_trace(go.Scatter(
                    x=t, y=drel[k],
                    mode="lines", name=f"{name} − G1",
                    line=dict(color=GEN_COLORS[k % len(GEN_COLORS)], width=2),
                    hovertemplate=f"{name}−G1<br>t=%{{x:.3f}} s  Δδ=%{{y:.1f}}°<extra></extra>",
                ))

            rfig.update_layout(
                xaxis_title="Time (s)",
                yaxis_title="δᵢ − δ_G1 (degrees)",
                height=360, plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"),
                margin=dict(l=10, r=10, t=10, b=40),
            )
            st.plotly_chart(rfig, use_container_width=True)

            # ── Critical clearing time ────────────────────────────────────────
            st.divider()
            st.markdown("**Critical clearing time (CCT)**")
            st.caption(
                "CCT is the maximum fault duration that keeps the system stable. "
                "This requires running ~10–20 simulations (bisection) and may take several seconds."
            )

            if st.button("🔎 Find CCT (bisection)", key="cct_btn"):
                with st.spinner("Searching for CCT…"):
                    try:
                        cct_res = find_cct(
                            buses, branches, stab_gens,
                            Vm, Va, Pg_solved, Qg_solved,
                            fault_bus=fault_bus,
                            drop_branch_idx=drop_idx,
                            t_lo=0.02, t_hi=1.5, tol=0.01,
                            t_end=t_end, baseMVA=100.0,
                            H_override=H_eff_arr,
                        )
                        st.session_state["cct_result"] = cct_res
                    except Exception as exc:
                        st.error(f"CCT search error: {exc}")

            cct_res = st.session_state.get("cct_result")
            if cct_res:
                cct_val = cct_res["cct"]
                st.info(cct_res["note"])
                cc1, cc2 = st.columns(2)
                cc1.metric("CCT", f"{cct_val:.3f} s",
                           delta=f"{cct_val * 60:.1f} cycles at 60 Hz")
                cc2.metric("Clearing time set", f"{t_clear:.3f} s",
                           delta="within CCT — STABLE" if t_clear <= cct_val else "exceeds CCT — UNSTABLE",
                           delta_color="off" if t_clear <= cct_val else "inverse")

        # ── CCT vs DER penetration sweep ──────────────────────────────────────
        st.divider()
        st.markdown("**CCT vs DER penetration — Legacy vs IEEE 1547-2018**")
        st.caption(
            "Bisects CCT at each penetration level for both DER models. "
            "Takes ~20–60 s for the 9-bus; longer for larger cases."
        )
        sweep_btn = st.button("📈 Run Penetration Sweep", key="sweep_btn")
        if sweep_btn:
            prog = st.progress(0.0, text="Running penetration sweep…")
            try:
                sweep_res = sweep_cct_vs_penetration(
                    buses, branches, stab_gens,
                    Vm, Va, Pg_solved, Qg_solved,
                    fault_bus=fault_bus,
                    drop_branch_idx=drop_idx,
                    penetration_levels=[0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
                    H_virtual=H_virt,
                    t_lo=0.02, t_hi=1.5, tol=0.02,
                    t_end=t_end, baseMVA=100.0,
                    progress_cb=lambda v: prog.progress(v, text=f"Sweep {v*100:.0f}%…"),
                )
                prog.empty()
                st.session_state["sweep_result"] = sweep_res
                st.session_state["sweep_case"]   = case_name
            except Exception as exc:
                prog.empty()
                st.error(f"Sweep error: {exc}")

        sw = st.session_state.get("sweep_result")
        if sw is not None and st.session_state.get("sweep_case") == case_name:
            pens_pct  = [p * 100 for p in sw["penetration"]]
            cct_leg   = sw["cct_legacy"]
            cct_1547  = sw["cct_1547"]
            H_leg     = sw["H_sys_legacy"]
            H_1547    = sw["H_sys_1547"]
            H_v       = sw["H_virtual"]

            sfig = go.Figure()
            sfig.add_trace(go.Scatter(
                x=pens_pct, y=cct_leg, mode="lines+markers", name="Legacy (pre-1547)",
                line=dict(color="#B71C1C", width=2.5, dash="dash"),
                marker=dict(size=8, color="#B71C1C"),
                hovertemplate="Legacy · %{x:.0f}% DER<br>CCT = %{y:.3f} s<extra></extra>",
            ))
            sfig.add_trace(go.Scatter(
                x=pens_pct, y=cct_1547, mode="lines+markers",
                name=f"IEEE 1547-2018 (H_v={H_v:.1f} s)",
                line=dict(color="#1565C0", width=2.5),
                marker=dict(size=8, color="#1565C0"),
                hovertemplate="1547-2018 · %{x:.0f}% DER<br>CCT = %{y:.3f} s<extra></extra>",
            ))
            # Typical relay clearing time reference lines
            sfig.add_hline(y=5/60,  line_dash="dot", line_color="#2E7D32",
                           annotation_text="5 cycles (fast relay)", annotation_position="top left")
            sfig.add_hline(y=20/60, line_dash="dot", line_color="#F57F17",
                           annotation_text="20 cycles (backup)", annotation_position="top left")

            sfig.update_layout(
                xaxis_title="DER penetration (%)",
                yaxis_title="Critical clearing time (s)",
                height=380, plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.9)"),
                margin=dict(l=10, r=10, t=30, b=40),
                title=dict(
                    text=f"CCT vs DER penetration — fault at Bus {sw['fault_bus']}  |  H_virtual = {H_v:.1f} s",
                    font=dict(size=12), x=0.5, xanchor="center",
                ),
            )
            st.plotly_chart(sfig, use_container_width=True)

            # Table
            sw_rows = []
            for i, p in enumerate(pens_pct):
                sw_rows.append({
                    "DER pen (%)": f"{p:.0f}",
                    "H_sys legacy (s)": f"{H_leg[i]:.2f}",
                    "CCT legacy (s)":   f"{cct_leg[i]:.3f}",
                    f"H_sys 1547 (s)":  f"{H_1547[i]:.2f}",
                    "CCT 1547 (s)":     f"{cct_1547[i]:.3f}",
                    "CCT gain (s)":     f"{cct_1547[i]-cct_leg[i]:+.3f}",
                })
            st.dataframe(pd.DataFrame(sw_rows), hide_index=True, use_container_width=True)

        # ── Explainer ─────────────────────────────────────────────────────────
        with st.expander("Understanding transient stability", expanded=False):
            st.markdown("""
**Classical machine model**

Each synchronous generator is represented as a constant internal voltage magnitude E' behind
a transient reactance X'd. The rotor angle δ evolves according to the **swing equation**:

> dδ/dt = Δω
> dΔω/dt = (ω_s / 2H) · (Pm − Pe)

- **δ** — rotor angle relative to a synchronously rotating reference frame
- **Δω** — angular frequency deviation (0 at synchronism)
- **H** — inertia constant [seconds]: higher H = harder to accelerate
- **Pm** — mechanical power input (constant: governor is slow relative to fault dynamics)
- **Pe** — electrical power output (drops sharply during the fault, recovers after clearing)

---

**What happens during a 3-phase fault?**

The faulted bus voltage collapses to (near) zero. The coupling between the faulted region and
the rest of the network decreases, so Pe drops. Since Pm stays constant, Pm − Pe > 0 and
the rotor accelerates. The longer the fault, the more the rotor angle advances.

If the angle advances too far before the fault is cleared, the restoring force post-fault is
insufficient to pull the rotor back into synchronism. This is **loss of synchronism** (pole slipping).

---

**Critical clearing time (CCT)**

The maximum fault duration that keeps all generators in step. Typical protection targets:
- High-speed relays: 3–5 cycles (0.05–0.083 s)
- Distance relays Zone 1: 4–6 cycles
- Backup clearing: 12–30 cycles (0.2–0.5 s)

A fault with clearing time beyond the CCT triggers cascading separation.

---

**Equal area criterion (for intuition)**

For a single-machine infinite-bus (SMIB) system, stability can be assessed graphically:
the *accelerating area* (energy gained during fault) must be smaller than the *decelerating area*
(maximum energy the system can absorb post-fault). The CCT corresponds to equal areas.

---

**Try this on the 9-bus case**
1. Fault at bus 7, clearing 0.10 s → **stable**
2. Increase clearing to 0.40 s → **unstable** — G3 (lowest H, 3.01 s) separates first
3. Find CCT → ~0.32 s at 0% DER penetration
4. Set DER to Legacy 50% → watch CCT drop; system H halves → rotor accelerates twice as fast
5. Switch to IEEE 1547-2018 at same 50% penetration → CCT recovers partially (virtual inertia)
6. Run the penetration sweep → see the diverging Legacy vs 1547-2018 curves

---

**DER penetration and the swing equation**

| Scenario | H_eff | ROCOF | CCT |
|---|---|---|---|
| 0% DER (100% synchronous) | H_orig | low | longest |
| Legacy DERs displace 50% | 0.5 × H_orig | 2× higher | much shorter |
| 1547-2018 DERs displace 50%, H_v = 3 s | 0.5 × H_orig + ½ × H_v | intermediate | partially recovered |

**Legacy DER behaviour during faults (IEEE 1547-2003)**
- Trip offline when V < 0.88 pu (most settings) — no low-voltage ride-through
- Suddenly remove generation exactly when the network needs stability support
- Anti-islanding protection fires within 2–10 cycles

**IEEE 1547-2018 improvements**
- **LVRT Category I**: stay connected down to 0.0 pu for 0.16 s, 0.45 pu for 0.32 s
- **LVRT Category III** (utility-scale): mirrors transmission interconnection requirements
- **Reactive current injection**: ΔI_q ≥ 2 × ΔV during voltage dip — supports voltage recovery
- **Grid-forming capability** (optional but growing): synthesises virtual inertia, provides short-circuit current, participates in frequency response
- ROCOF ride-through: must not trip on ROCOF ≤ 2 Hz/s (Category I) or ≤ 3 Hz/s (Cat. III)

**Why this matters for your protection engineers**
- Lower fault current from inverter-based DERs reduces relay sensitivity → longer clearing time
- Longer clearing time + shorter CCT = narrowing stability margin
- Distance relays may underreach (apparent impedance shifts with DER infeed)
- ROCOF relays must be recoordinated — settings calibrated for high-H systems will false-trip
- Anti-islanding and LVRT requirements can conflict → new protection philosophies needed
            """)
