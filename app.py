"""
Web app for the HT-MPC vs RT-MPC lane-change playground.

    streamlit run app.py

The method is NOT reimplemented here: the cell of `LatVeh_Interactive.ipynb` that
holds it (the verbatim copy of `LatVeh_Benchmark.ipynb` plus design/run/animate)
is executed as-is at start-up, so the app, the notebook and the paper's benchmark
can never drift apart.
"""
import json

import pandas as pd
import streamlit as st

NOTEBOOK = "LatVeh_Interactive.ipynb"
MENU_T_SIM = {"Single lane change (paper)": 5.0, "Slalom": 9.0, "Custom waypoints": 5.0}

st.set_page_config(page_title="Homothetic Tube MPC - lane change", page_icon="🚗",
                   layout="wide")


@st.cache_resource(show_spinner="Loading the method from the notebook ...")
def load_method():
    """Execute the notebook's method cell and hand back its namespace."""
    nb = json.load(open(NOTEBOOK, encoding="utf-8"))
    src = next("".join(c["source"]) for c in nb["cells"]
               if c["cell_type"] == "code" and "def animate(" in "".join(c["source"]))
    ns = {}
    exec(compile(src, f"{NOTEBOOK}#method", "exec"), ns)
    return ns


M = load_method()


@st.cache_resource(show_spinner="Offline design: RPI ellipsoid, tightening, terminal set ...")
def get_design(v_kmh, dv_max, w_ey):
    return M["design"](v_kmh=v_kmh, dv_max=dv_max, w_ey=w_ey)


# cache_resource, not cache_data: the results are dataclasses defined in the
# notebook's namespace, which pickle cannot serialise
@st.cache_resource(show_spinner=False, max_entries=64)
def run_case(v_kmh, dv_max, w_ey, mode, T_sim, custom):
    return M["run"](get_design(v_kmh, dv_max, w_ey), mode, T_sim=T_sim, custom=custom)


DV_PAPER = float(M["DV_MAX_DEFAULT"])
W_PAPER = float(M["W_MAX_DEFAULT"][0])
DV_LEVELS = sorted({0.05, 0.10, 0.15, 0.20, DV_PAPER, 0.30, 0.35, 0.40, 0.45})
W_LEVELS = sorted({0.005, 0.010, 0.020, 0.030, 0.040, W_PAPER, 0.060, 0.065})
_fmt = lambda v, ref: f"{v:.3f} (paper)" if abs(v - ref) < 1e-9 else f"{v:.3f}"

# ── Sidebar: the controls ────────────────────────────────────────────────────
with st.sidebar:
    st.title("Controls")
    mode = st.selectbox("Manoeuvre", list(MENU_T_SIM), index=0)
    T_sim = st.slider("T_sim - simulation length [s]", 2.0, 15.0,
                      MENU_T_SIM[mode], 0.5, key=f"tsim_{mode}")
    custom = ""
    if mode == "Custom waypoints":
        n_seg = int(st.number_input("segments", 2, 12, 4, step=1))
        st.caption("One column per segment: its duration [s] and the lane position e_y [m] "
                   "it holds. The last duration is filled automatically to add up to T_sim.")
        _y0 = [-1.75, 1.75, 0.0, -1.75, 1.75, 0.0, -1.75, 1.75, 0.0, -1.75, 1.75, 0.0]
        base = pd.DataFrame({f"seg {i+1}": [round(T_sim/n_seg, 2), _y0[i]] for i in range(n_seg)},
                            index=["duration [s]", "e_y [m]"])
        tab = st.data_editor(base, key=f"segtab_{n_seg}_{T_sim}")
        dur = [float(tab.iloc[0, i]) for i in range(n_seg - 1)]
        rest = round(T_sim - sum(dur), 2)
        if rest < 0.1:
            st.error(f"The first {n_seg-1} durations exceed T_sim = {T_sim:.1f} s.")
            st.stop()
        starts = [0.0] + list(pd.Series(dur).cumsum())
        st.caption(f"last segment {rest:.1f} s   (starts: " +
                   ", ".join(f"{t:.1f}" for t in starts) + ")")
        custom = "; ".join(f"{starts[i]:.3f}:{float(tab.iloc[1, i])}" for i in range(n_seg))

    st.divider()
    st.caption("**Uncertainty bounds** (Assumption 1 of the paper)")
    v_kmh = st.slider("speed v_x0 [km/h]", 40, 95, 60, 5)
    dv_max = st.select_slider("Δv_x bound [m/s]", DV_LEVELS, DV_PAPER,
                              format_func=lambda v: _fmt(v, DV_PAPER))
    w_ey = st.select_slider("w bound on e_y [m]", W_LEVELS, W_PAPER,
                            format_func=lambda v: _fmt(v, W_PAPER))
    st.divider()
    panels = st.checkbox("also animate α_k and δ_s", False)
    dark = st.checkbox("dark background", False)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("Homothetic tube MPC with closed-form scaling — lane change")
st.markdown(
    "HT-MPC (proposed) and RT-MPC (rigid tube) solve the **same QP**, with the same RPI "
    "ellipsoid $\\Omega$, the same terminal cost and the same terminal set. The only "
    "difference is the tightening: HT-MPC scales it by the measured "
    "$\\alpha_k=\\|e_k\\|_{P^{\\star-1}}$, RT-MPC always uses $\\alpha_{j|k}\\equiv1$. "
    "Move a control and both closed loops are re-simulated; press **▶ Play** on the figure."
)

# ── Design + run ─────────────────────────────────────────────────────────────
try:
    d = get_design(v_kmh, dv_max, w_ey)
except ValueError as e:
    st.error(f"**No design exists for these bounds.** {e}")
    st.info("Assumption 3 fails: Ω no longer fits inside the constraints. "
            "Lower the speed or one of the two bounds.")
    st.stop()

try:
    res_p, res_r, y_ref, dvx = run_case(v_kmh, dv_max, w_ey, mode, T_sim, custom)
except ValueError as e:
    st.error(f"Invalid manoeuvre: {e}")
    st.stop()

sp, sr = M["summary"](res_p), M["summary"](res_r)

cols = st.columns(5)
cols[0].metric("HT-MPC", sp["status"], f"cost {sp['cost']:.0f}", delta_color="off")
cols[1].metric("RT-MPC", sr["status"], f"cost {sr['cost']:.0f}", delta_color="off")
cols[2].metric("cost reduction",
               f"{100*(1-sp['cost']/sr['cost']):.1f} %" if (sp["ok"] and sr["ok"]) else "—",
               help="closed-loop cost of HT-MPC against RT-MPC on the same realisation")
cols[3].metric("max |δ| HT / RT [rad]", f"{sp['max_delta']:.3f} / {sr['max_delta']:.3f}",
               f"rigid margin {d.u_margin:.3f}", delta_color="off")
cols[4].metric("Ω half-width in e_y [m]", f"{d.s_ey:.3f}",
               f"Δv_x drawn {dvx:+.3f} m/s", delta_color="off")

if sp["ok"] and not sr["ok"]:
    st.success("**RT-MPC's QP is infeasible here, HT-MPC still completes the manoeuvre.** "
               "The rigid tightening has eaten the steering budget; the homothetic one "
               "scales with the error that is actually there.")
elif not sp["ok"]:
    st.warning("Both schemes are infeasible at these bounds — the lane change no longer fits "
               "inside the tightened constraints.")

fig = M["animate"](d, res_p, res_r, y_ref,
                   title=f"{mode} @ {v_kmh} km/h",
                   frame_ms=int(round(1000*d.Ts)), panels=panels,
                   theme=("dark" if dark else "light"))
st.plotly_chart(fig, theme=None, config={"displaylogo": False})

with st.expander("What am I looking at?"):
    st.markdown(
        """
* The figure is the paper's lane-change figure, drawn progressively: the two trajectories
  $e_y(x)$ with the cross-sections each scheme uses to tighten the **current** step —
  $\\alpha_k\\Omega$ for HT-MPC (thin, because the true error stays small) and the full
  $\\Omega$ for RT-MPC.
* **The bounds** (Assumption 1) are the speed error $|\\Delta v_x|\\le\\overline{\\Delta v}$ and
  the additive disturbance $w_k\\in\\mathrm{Ell}(\\Gamma_w)$, set here by its $e_y$ semi-axis.
  Larger bounds give a larger $\\Omega$, more tightening and less steering left to the nominal
  input: at $\\overline{\\Delta v}=0.35$ m/s, at 45 km/h, above 85 km/h or at
  $\\bar w_{e_y}=0.060$ m, RT-MPC becomes infeasible while HT-MPC still drives the manoeuvre.
* With the default (paper) bounds this reproduces the benchmark exactly: closed-loop cost
  1481.0 against 2238.1, a 33.8 % reduction.
* Everything runs on open-source solvers (Clarabel, OSQP, SCS through CVXPY).
        """
    )
st.caption("Method loaded from `LatVeh_Interactive.ipynb`; the same code produces the figure "
           "of the paper in `LatVeh_Benchmark.ipynb`.")
