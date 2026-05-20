# AC Power Systems Explorer

An interactive power systems analysis tool built with Python and Streamlit. Runs a full Newton–Raphson AC power flow on IEEE standard test cases, then layers on DC-OPF with LMPs, N-1 contingency analysis, transient stability simulation, and economic dispatch — all in the browser.

## Features

- **AC Power Flow** — Newton–Raphson solver with DC warm start, PV→PQ warm-up, backtracking line search, and line-charging homotopy for convergence on ill-conditioned cases
- **DC Optimal Power Flow (OPF)** — linear program minimizing generation cost subject to generator output and thermal limits; dual variables give Locational Marginal Prices (LMPs)
- **N-1 Contingency Analysis** — removes each branch one at a time and re-solves; flags voltage violations and thermal overloads in the surviving network
- **Transient Stability** — classical constant-E′ machine model integrated with `scipy.solve_ivp`; includes fault simulation, fault clearing, and optional post-fault branch removal
- **DER Penetration** — models the effect of displacing synchronous generation with inverter-based DERs on system inertia and ROCOF
- **Economic Dispatch** — separable page with interactive cost curves and marginal cost analysis

## Test Cases

| Case | Buses | Type | Notes |
|---|---|---|---|
| 3-Bus Example | 3 | Transmission | Simple pedagogical case |
| IEEE 9-Bus (WSCC) | 9 | Transmission | Anderson & Fouad reference |
| IEEE 14-Bus | 14 | Transmission | MATPOWER case14; 3 off-nominal transformers |
| IEEE 39-Bus (New England) | 39 | Transmission | 10 generators; 46 branches |
| IEEE 33-Bus (Baran & Wu) | 33 | Distribution | 12.66 kV radial feeder |
| IEEE 69-Bus (Das et al.) | 69 | Distribution | 12.66 kV radial; severe undervoltage at bus 65 |

## Project Structure

```
acpf_nr.py          # NR solver, Ybus, Jacobian, branch flows, IEEE test cases
app.py              # Main Streamlit app (network diagram, OPF, N-1, stability tabs)
opf.py              # DC OPF with LMP via scipy.optimize.linprog
stability.py        # Swing equation integration; DER inertia modelling
dispatch.py         # Economic dispatch logic
pages/
  2_Economic_Dispatch.py   # Streamlit page for economic dispatch
requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

To run the solver standalone and export CSV + PNG snapshots:

```bash
python acpf_nr.py
```

## Requirements

- Python 3.11+
- streamlit ≥ 1.32
- plotly ≥ 5.20
- numpy ≥ 1.26
- pandas ≥ 2.0
- scipy ≥ 1.12

## Solver Details

The Newton–Raphson implementation in `acpf_nr.py`:

- Builds the full complex **Ybus** with π-model lines, bus shunts, and off-nominal transformer taps
- Uses a **DC power flow** to warm-start voltage angles before the first NR iteration
- Treats PV buses as PQ for the first 5 iterations (warm-up) to improve convergence on cases with tight voltage setpoints
- Assembles the **4-quadrant Jacobian** (H/N/M/L) analytically
- Applies a **trust-region cap** (±15° on angles, ±0.15 pu on voltage magnitude) before each step
- Uses **backtracking line search** (Armijo-style) to ensure the mismatch decreases each iteration
- Enforces **Q-limits** on PV buses and converts violating buses to PQ mid-solve
- Ramps line-charging susceptance from 0 → 1 in 5 homotopy stages for robustness

Convergence tolerance defaults to 1×10⁻⁶ pu on the infinity-norm of the power mismatch.
