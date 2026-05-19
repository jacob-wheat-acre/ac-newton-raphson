import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from dispatch import (
    DispatchGen, FuelType, FUEL_LABELS, FUEL_COLORS, THERMAL_FUELS,
    default_fleet, economic_dispatch,
)


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert a '#rrggbb' hex string to 'rgba(r,g,b,alpha)' for Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

st.set_page_config(page_title="Economic Dispatch", page_icon="⚡", layout="wide")
st.title("Economic Dispatch & Generator Thermodynamics")
st.markdown(
    "Power plants are committed and loaded in **merit order** — cheapest first — "
    "to supply a given load at minimum total operating cost. "
    "This page shows the heat-rate thermodynamics, the supply stack, and the optimal dispatch."
)
st.divider()

# ── Background ────────────────────────────────────────────────────────────────
with st.expander("Background: from steam to megawatts", expanded=False):
    st.markdown("""
### The Rankine cycle and heat rate

Most thermal generators (coal, nuclear, gas combined-cycle) convert heat to work via the
**Rankine (steam) cycle**:

```
  Fuel combustion / nuclear fission
         ↓   heat Q_in
  Boiler/steam generator  →  Turbine  →  Generator  →  Grid (W_out)
                                ↓
                            Condenser  →  Q_reject to cooling water
```

The **thermal efficiency** is limited by the Carnot bound:

> **η_Carnot = 1 − T_cold / T_hot**

For a coal plant with T_hot ≈ 550 °C (823 K) and T_cold ≈ 30 °C (303 K):
η_Carnot ≈ 63%.  Real plants achieve 35–55% due to irreversibilities.

### Heat rate H(P)

The **heat rate** (or *heat input curve*) H(P) [MBtu/hr] is the total fuel energy consumed
to produce P MW of electricity.  Divide by P to get the *specific* heat rate:

> **HR = H(P) / P  [MBtu/MWh]**

Since 1 MWh ≡ 3.412 MBtu of electrical energy, efficiency is:

> **η = 3.412 / HR**

A quadratic model is standard:

> **H(P) = a + b·P + c·P²**

where *a* is the no-load heat (keeping the boiler warm), *b* is the incremental heat rate
near rated output, and *c* captures the rise in heat rate at very high loading.

### Economic dispatch

With N generators and a total load P_D, the **equal incremental cost** criterion says:
at the optimum, all *marginal costs* are equal:

> **λ = dC₁/dP₁ = dC₂/dP₂ = … = dCₙ/dPₙ**

Here λ [$/MWh] is the *system lambda* — the marginal cost of serving one more MWh of load.
The merit order is simply the ranking by marginal cost from cheapest to most expensive.
""")
    st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Fleet settings")

    st.subheader("Demand")
    Pload = st.slider("Total load demand (MW)", 710, 2250, 1400, 25,
                      help="Minimum = sum of all Pmin; maximum = sum of all Pmax")

    st.subheader("Fuel prices")
    gas_price = st.slider("Natural gas ($/MBtu)", 2.0, 14.0, 5.50, 0.25,
                          help="Applies to both CCGT and Gas CT")
    coal_price = st.slider("Coal ($/MBtu)", 1.0, 5.0, 2.50, 0.25)

    st.subheader("Available capacity (%)")
    avail = {}
    for ft in [FuelType.NUCLEAR, FuelType.COAL, FuelType.GAS_CC,
               FuelType.GAS_CT, FuelType.HYDRO, FuelType.WIND, FuelType.SOLAR]:
        default = 80 if ft == FuelType.SOLAR else (85 if ft == FuelType.WIND else 100)
        avail[ft] = st.slider(FUEL_LABELS[ft], 0, 100, default,
                              help="% of rated Pmax available (weather, outage, water, etc.)")

# ── Build adjusted fleet ───────────────────────────────────────────────────────
fleet = default_fleet()
for g in fleet:
    if g.fuel in (FuelType.GAS_CC, FuelType.GAS_CT):
        g.fuel_cost = gas_price
    if g.fuel == FuelType.COAL:
        g.fuel_cost = coal_price
    g.Pmax = g.Pmax * avail[g.fuel] / 100.0

# ── Solve economic dispatch ───────────────────────────────────────────────────
P_opt, total_cost, ok, msg = economic_dispatch(fleet, Pload)

if not ok:
    st.error(f"Economic dispatch failed: {msg}")
    st.stop()

n_gens = len(fleet)
lambda_sys = max(g.mc(np.array([P_opt[i]]))[0] for i, g in enumerate(fleet) if P_opt[i] > fleet[i].Pmin + 0.5)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Generator Curves", "Merit Order & Dispatch", "CO₂ & Emissions"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Thermodynamic curves (thermal generators only)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Thermodynamic curves — thermal generators")
    st.markdown(
        "These curves only apply to thermal generators (nuclear, coal, gas). "
        "Renewables and hydro have no combustion heat input."
    )

    thermal = [g for g in fleet if g.fuel in THERMAL_FUELS]

    fig_th = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Heat rate H(P) [MBtu/hr]",
            "Specific heat rate [MBtu/MWh]",
            "Thermal efficiency η(%)",
            "Marginal cost dC/dP [$/MWh]",
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    for g in thermal:
        P_arr = np.linspace(max(g.Pmin, 1), g.Pmax, 300)
        color = FUEL_COLORS[g.fuel]

        # Heat input
        fig_th.add_trace(go.Scatter(
            x=P_arr, y=g.H(P_arr), name=g.name,
            line=dict(color=color, width=2),
            legendgroup=g.name, showlegend=True,
        ), row=1, col=1)

        # Specific heat rate
        fig_th.add_trace(go.Scatter(
            x=P_arr, y=g.HR(P_arr), name=g.name,
            line=dict(color=color, width=2),
            legendgroup=g.name, showlegend=False,
        ), row=1, col=2)

        # Efficiency
        fig_th.add_trace(go.Scatter(
            x=P_arr, y=g.eta(P_arr) * 100, name=g.name,
            line=dict(color=color, width=2),
            legendgroup=g.name, showlegend=False,
        ), row=2, col=1)

        # Marginal cost
        fig_th.add_trace(go.Scatter(
            x=P_arr, y=g.mc(P_arr), name=g.name,
            line=dict(color=color, width=2),
            legendgroup=g.name, showlegend=False,
        ), row=2, col=2)

        # Mark dispatch operating point
        pi = P_opt[fleet.index(g)]
        if pi > 1:
            for (r, c), y_val in [
                ((1, 1), g.H(np.array([pi]))[0]),
                ((1, 2), g.HR(np.array([pi]))[0]),
                ((2, 1), g.eta(np.array([pi]))[0] * 100),
                ((2, 2), g.mc(np.array([pi]))[0]),
            ]:
                fig_th.add_trace(go.Scatter(
                    x=[pi], y=[y_val], mode="markers",
                    marker=dict(color=color, size=10, symbol="circle",
                                line=dict(color="white", width=1.5)),
                    name=f"{g.name} dispatch",
                    legendgroup=g.name, showlegend=False,
                    hovertemplate=f"<b>{g.name}</b><br>P = {pi:.1f} MW<br>y = {y_val:.3f}<extra></extra>",
                ), row=r, col=c)

    # System lambda line on MC plot
    fig_th.add_hline(y=lambda_sys, row=2, col=2,
                     line=dict(color="red", dash="dash", width=1.5),
                     annotation_text=f"λ = ${lambda_sys:.2f}/MWh",
                     annotation_font_color="red")

    fig_th.update_xaxes(title_text="P (MW)", row=2, col=1)
    fig_th.update_xaxes(title_text="P (MW)", row=2, col=2)
    fig_th.update_yaxes(title_text="MBtu/hr", row=1, col=1)
    fig_th.update_yaxes(title_text="MBtu/MWh", row=1, col=2)
    fig_th.update_yaxes(title_text="η (%)", row=2, col=1)
    fig_th.update_yaxes(title_text="$/MWh", row=2, col=2)
    fig_th.update_layout(height=580, plot_bgcolor="white", paper_bgcolor="white",
                         legend=dict(x=1.01, y=0.5))

    st.plotly_chart(fig_th, use_container_width=True)

    st.caption(
        "Filled circles = operating point from the economic dispatch solution.  "
        "Red dashed line = system lambda λ (marginal cost of the last MW dispatched)."
    )

    # Efficiency summary table
    eff_rows = []
    for g in thermal:
        pi = P_opt[fleet.index(g)]
        if pi > 0.5:
            hr_val = g.HR(np.array([pi]))[0]
            eta_val = g.eta(np.array([pi]))[0]
            eff_rows.append({
                "Generator": g.name,
                "Dispatch (MW)": f"{pi:.1f}",
                "Heat rate (MBtu/MWh)": f"{hr_val:.3f}",
                "Efficiency (%)": f"{eta_val * 100:.1f}",
                "MC ($/MWh)": f"{g.mc(np.array([pi]))[0]:.2f}",
                "Cost ($/hr)": f"{g.cost(np.array([pi]))[0]:,.0f}",
            })
    if eff_rows:
        st.dataframe(pd.DataFrame(eff_rows), hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Merit order supply stack + dispatch result
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Supply stack (merit order)")
        st.markdown(
            "Each block is one generator: width = available capacity, "
            "height = marginal cost at mid-output. "
            "The vertical red line is today's load demand."
        )

        # Sort by MC at midpoint
        mc_mid = [g.mc(np.array([(g.Pmin + g.Pmax) / 2.0]))[0] for g in fleet]
        order = sorted(range(n_gens), key=lambda i: mc_mid[i])

        fig_stack = go.Figure()
        cumulative = 0.0
        for i in order:
            g = fleet[i]
            cap = g.Pmax
            mc_val = mc_mid[i]
            color = FUEL_COLORS[g.fuel]

            # Draw block as a filled scatter
            fig_stack.add_trace(go.Scatter(
                x=[cumulative, cumulative + cap, cumulative + cap, cumulative, cumulative],
                y=[0, 0, mc_val, mc_val, 0],
                fill="toself",
                fillcolor=_rgba(color, 0.6),
                line=dict(color=color, width=1.5),
                name=f"{FUEL_LABELS[g.fuel]} ({g.name})",
                hovertemplate=(
                    f"<b>{g.name}</b><br>"
                    f"Fuel: {FUEL_LABELS[g.fuel]}<br>"
                    f"Capacity: {cap:.0f} MW<br>"
                    f"MC (mid): ${mc_val:.2f}/MWh<br>"
                    f"Dispatch: {P_opt[i]:.1f} MW"
                    "<extra></extra>"
                ),
            ))
            cumulative += cap

        # Demand line
        fig_stack.add_vline(
            x=Pload,
            line=dict(color="red", dash="dash", width=2),
            annotation_text=f"Demand {Pload} MW",
            annotation_font_color="red",
        )
        # Lambda line
        fig_stack.add_hline(
            y=lambda_sys,
            line=dict(color="darkred", dash="dot", width=1.5),
            annotation_text=f"λ = ${lambda_sys:.2f}/MWh",
            annotation_font_color="darkred",
            annotation_position="bottom right",
        )

        fig_stack.update_layout(
            xaxis=dict(title="Cumulative capacity (MW)", showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(title="Marginal cost ($/MWh)", showgrid=True, gridcolor="#f0f0f0"),
            plot_bgcolor="white", paper_bgcolor="white",
            height=400,
            legend=dict(x=0.01, y=0.99, font=dict(size=10),
                        bgcolor="rgba(255,255,255,0.9)"),
            margin=dict(l=20, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_stack, use_container_width=True)

    with col_right:
        st.subheader("Dispatch result")

        # Bar chart by generator
        sorted_by_fuel = sorted(range(n_gens), key=lambda i: mc_mid[i])
        gen_names = [fleet[i].name for i in sorted_by_fuel]
        gen_P     = [P_opt[i] for i in sorted_by_fuel]
        gen_colors = [FUEL_COLORS[fleet[i].fuel] for i in sorted_by_fuel]

        fig_bar = go.Figure(go.Bar(
            x=gen_names, y=gen_P,
            marker_color=gen_colors,
            text=[f"{p:.1f}" for p in gen_P],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Dispatch: %{y:.1f} MW<extra></extra>",
        ))
        fig_bar.update_layout(
            yaxis_title="Dispatch (MW)",
            plot_bgcolor="white", paper_bgcolor="white",
            height=350,
            margin=dict(l=10, r=10, t=10, b=60),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Key metrics ──────────────────────────────────────────────────────────
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total cost", f"${total_cost:,.0f}/hr",
              help="Total fuel + variable O&M for current dispatch")
    c2.metric("System λ", f"${lambda_sys:.2f}/MWh",
              help="Marginal cost of the last MW — price signal for dispatch")
    c3.metric("Avg cost", f"${total_cost/max(Pload, 1):.2f}/MWh",
              help="Total cost / total generation")
    c4.metric("Available capacity", f"{sum(g.Pmax for g in fleet):.0f} MW",
              help="Sum of all generator Pmax at current availability settings")

    # ── Dispatch table ───────────────────────────────────────────────────────
    st.subheader("Dispatch summary")
    disp_rows = []
    for i in sorted_by_fuel:
        g = fleet[i]
        pi = P_opt[i]
        pct = 100 * pi / g.Pmax if g.Pmax > 0 else 0
        cost_hr = g.cost(np.array([pi]))[0]
        mc_val  = g.mc(np.array([pi]))[0]
        disp_rows.append({
            "Generator":     g.name,
            "Fuel":          FUEL_LABELS[g.fuel],
            "Pmin (MW)":     f"{g.Pmin:.0f}",
            "Pmax (MW)":     f"{g.Pmax:.0f}",
            "Dispatch (MW)": f"{pi:.1f}",
            "Loading (%)":   f"{pct:.0f}%",
            "MC ($/MWh)":    f"{mc_val:.2f}",
            "Cost ($/hr)":   f"${cost_hr:,.0f}",
        })
    st.dataframe(pd.DataFrame(disp_rows), hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CO₂ emissions
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("CO₂ emissions by generator")
    st.markdown("""
    Emission factors use EPA eGRID defaults for combustion CO₂:
    - Coal: **0.0948 ton CO₂/MBtu**
    - Natural gas (CCGT & CT): **0.0531 ton CO₂/MBtu**
    - Nuclear, hydro, wind, solar: **0 ton CO₂/MBtu** (no combustion)
    """)

    co2_rows = []
    total_co2_hr = 0.0
    for i in range(n_gens):
        g = fleet[i]
        pi = P_opt[i]
        co2_hr = g.co2_per_hr(np.array([pi]))[0]
        co2_mwh = g.co2_per_mwh(np.array([pi]))[0] if pi > 0.5 else 0.0
        total_co2_hr += co2_hr
        co2_rows.append({
            "Generator":    g.name,
            "Fuel":         FUEL_LABELS[g.fuel],
            "Dispatch (MW)": f"{pi:.1f}",
            "CO₂ (ton/hr)": f"{co2_hr:.2f}",
            "CO₂ intensity (ton/MWh)": f"{co2_mwh:.4f}" if pi > 0.5 else "—",
        })

    total_intensity = total_co2_hr / max(Pload, 1)

    m1, m2, m3 = st.columns(3)
    m1.metric("Total CO₂", f"{total_co2_hr:.1f} ton/hr")
    m2.metric("Grid intensity", f"{total_intensity:.4f} ton/MWh",
              help="Fleet-average CO₂ per MWh at current dispatch")
    m3.metric("CO₂-free generation",
              f"{sum(P_opt[i] for i in range(n_gens) if fleet[i].fuel not in {FuelType.COAL, FuelType.GAS_CC, FuelType.GAS_CT}):.0f} MW")

    st.divider()

    col_em1, col_em2 = st.columns([1, 1])

    with col_em1:
        # Stacked bar by fuel: MW dispatched, colored by fuel type
        fuels_in_use = [g.fuel for g in fleet if P_opt[fleet.index(g)] > 0.5]
        seen = set()
        fig_pie = go.Figure()
        labels = [g.name for g in fleet if P_opt[fleet.index(g)] > 0.5]
        values_mw = [P_opt[fleet.index(g)] for g in fleet if P_opt[fleet.index(g)] > 0.5]
        colors_pie = [FUEL_COLORS[g.fuel] for g in fleet if P_opt[fleet.index(g)] > 0.5]
        fig_pie.add_trace(go.Pie(
            labels=labels, values=values_mw,
            marker=dict(colors=colors_pie),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value:.1f} MW<br>%{percent}<extra></extra>",
        ))
        fig_pie.update_layout(title="Generation mix (MW)", height=350,
                              margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_em2:
        # CO2 by generator bar chart
        co2_gens = [g for g in fleet if P_opt[fleet.index(g)] > 0.5 and g.fuel in {FuelType.COAL, FuelType.GAS_CC, FuelType.GAS_CT}]
        if co2_gens:
            fig_co2 = go.Figure(go.Bar(
                x=[g.name for g in co2_gens],
                y=[g.co2_per_hr(np.array([P_opt[fleet.index(g)]]))[0] for g in co2_gens],
                marker_color=[FUEL_COLORS[g.fuel] for g in co2_gens],
                text=[f"{g.co2_per_hr(np.array([P_opt[fleet.index(g)]]))[0]:.1f}" for g in co2_gens],
                textposition="auto",
            ))
            fig_co2.update_layout(
                yaxis_title="CO₂ (ton/hr)", title="Emissions by generator",
                plot_bgcolor="white", paper_bgcolor="white",
                height=350, margin=dict(l=10, r=10, t=40, b=40),
            )
            st.plotly_chart(fig_co2, use_container_width=True)
        else:
            st.success("Zero direct CO₂ emissions at current dispatch — all generation from nuclear, hydro, wind, or solar.")

    st.dataframe(pd.DataFrame(co2_rows), hide_index=True, use_container_width=True)

    st.caption("""
    Note: lifecycle emissions (construction, fuel mining, waste) are not included here.
    Nuclear has near-zero lifecycle CO₂ (~12 g/kWh); wind ~11 g/kWh; solar ~40 g/kWh.
    These are combustion-only figures from EPA eGRID.
    """)
