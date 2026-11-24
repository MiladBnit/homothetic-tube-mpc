"""
Web app for the HT-MPC vs RT-MPC lane-change playground.

    streamlit run app.py

The method is NOT reimplemented here: the cell of `LatVeh_Interactive.ipynb` that
holds it (the verbatim copy of `LatVeh_Benchmark.ipynb` plus design/run/animate)
is executed as-is at start-up, so the app, the notebook and the paper's benchmark
can never drift apart.
"""
import hashlib
import json

import pandas as pd
import streamlit as st

NOTEBOOK = "LatVeh_Interactive.ipynb"
MENU_T_SIM = {"Single lane change (paper)": 5.0, "Slalom": 9.0, "Custom waypoints": 5.0}

st.set_page_config(page_title="Homothetic Tube MPC - lane change", page_icon="🚗",
                   layout="wide")


def _method_source():
    """The notebook's method cell (the code cell that defines `animate`)."""
    nb = json.load(open(NOTEBOOK, encoding="utf-8"))
    return next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "def animate(" in "".join(c["source"]))


# Every cache below is keyed by the hash of the method cell.  Streamlit Cloud
# reloads app.py on a push without restarting the process, so a cache keyed by
# the arguments alone would keep serving the code (and the designs) of the
# previous version of the notebook.
SRC = _method_source()
KEY = hashlib.sha256(SRC.encode("utf-8")).hexdigest()[:16]


@st.cache_resource(show_spinner="Loading the method from the notebook ...", max_entries=2)
def load_method(key):
    """Execute the notebook's method cell and hand back its namespace."""
    ns = {}
    exec(compile(SRC, f"{NOTEBOOK}#method", "exec"), ns)
    return ns


M = load_method(KEY)


@st.cache_resource(show_spinner="Offline design: RPI ellipsoid, tightening, terminal set ...")
def get_design(key, v_kmh, dvx_max_kmh, w_ey):
    return M["design"](v_kmh=v_kmh, dvx_max_kmh=dvx_max_kmh, w_ey=w_ey)


# cache_resource, not cache_data: the results are dataclasses defined in the
# notebook's namespace, which pickle cannot serialise
@st.cache_resource(show_spinner=False, max_entries=64)
def run_case(key, v_kmh, dvx_max_kmh, w_ey, mode, T_sim, custom):
    return M["run"](get_design(key, v_kmh, dvx_max_kmh, w_ey), mode, T_sim=T_sim,
                    custom=custom)


R2D = float(M["R2D"])
V_PAPER = int(M["V_KMH_DEFAULT"])
DV_PAPER = float(M["DVX_KMH_DEFAULT"])
W_PAPER = float(M["W_MAX_DEFAULT"][0])
DV_LEVELS = sorted({0.25, 0.50, 0.75, DV_PAPER, 1.25, 1.50, 1.75})
W_LEVELS = sorted({0.005, 0.010, 0.020, 0.030, 0.040, W_PAPER, 0.060, 0.065, 0.070})
_fmt = lambda v, ref, f=".3f": f"{v:{f}} (paper)" if abs(v - ref) < 1e-9 else f"{v:{f}}"

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
    v_kmh = st.slider("speed v_x0 [km/h]", 40, 95, V_PAPER, 5)
    dvx_max_kmh = st.select_slider("Δv_x bound [km/h]", DV_LEVELS, DV_PAPER,
                                   format_func=lambda v: _fmt(v, DV_PAPER, ".2f"),
                                   help="the simulated vehicle drives at the worst case "
                                        "v_x0 + bound, as in the paper")
    w_ey = st.select_slider("w bound on e_y [m]", W_LEVELS, W_PAPER,
                            format_func=lambda v: _fmt(v, W_PAPER))
    st.divider()
    dark = st.checkbox("dark background", False)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("Homothetic Tube MPC at the Computational Cost of Rigid Tube MPC")
st.caption("Interactive lane-change playground for the paper by M. Banitalebi Dehkordi, "
           "M. Mejari and D. Piga")
st.markdown(
    "HT-MPC (proposed) and RT-MPC (rigid tube) solve the **same QP**, with the same RPI "
    "ellipsoid $\\Omega$, the same terminal cost and the same terminal set. The only "
    "difference is the tightening: HT-MPC scales it by the measured "
    "$\\alpha_k=\\|e_k\\|_{P^{\\star-1}}$, RT-MPC always uses $\\alpha_{j|k}\\equiv1$. "
    "Move a control and both closed loops are re-simulated; press **▶ Play** on the figure."
)

# ── Design + run ─────────────────────────────────────────────────────────────
try:
    d = get_design(KEY, v_kmh, dvx_max_kmh, w_ey)
except ValueError as e:
    st.error(f"**No design exists for these bounds.** {e}")
    st.info("Assumption 3 fails: Ω no longer fits inside the constraints. "
            "Lower the speed or one of the two bounds.")
    st.stop()

try:
    res_p, res_r, y_ref, dvx_kmh = run_case(KEY, v_kmh, dvx_max_kmh, w_ey, mode, T_sim, custom)
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
cols[3].metric("max |v₀|ₖ| HT / RT [deg]", f"{sp['max_v0_deg']:.2f} / {sr['max_v0_deg']:.2f}",
               f"rigid bound {d.u_margin*R2D:.2f}°", delta_color="off",
               help="largest nominal steering each scheme used; RT-MPC may never exceed "
                    "the rigid bound")
cols[4].metric("Ω half-width in e_y [m]", f"{d.s_ey:.3f}",
               f"plant Δv_x {dvx_kmh:+.2f} km/h", delta_color="off")

if sp["ok"] and not sr["ok"]:
    st.success("**RT-MPC's QP is infeasible here, HT-MPC still completes the manoeuvre.** "
               "The rigid tightening has eaten the steering budget; the homothetic one "
               "scales with the error that is actually there.")
elif not sp["ok"]:
    st.warning("Both schemes are infeasible at these bounds — the lane change no longer fits "
               "inside the tightened constraints.")

fig = M["animate"](d, res_p, res_r, y_ref,
                   title=f"{mode} @ {v_kmh} km/h",
                   frame_ms=int(round(1000*d.Ts)),
                   theme=("dark" if dark else "light"))
st.plotly_chart(fig, theme=None, config={"displaylogo": False})

with st.expander("What am I looking at?"):
    st.markdown(
        """
* The figure is the paper's lane-change figure, drawn progressively. **Top:** the lateral
  position $e_y$ of both schemes. **Middle:** the nominal steering $v_{0|k}$ and the set
  each scheme admits for it at the **current** step — $H_u v\\le h_u-\\alpha_k s_u$ for
  HT-MPC (wide, because the true error stays small) and $H_u v\\le h_u-s_u$ for RT-MPC — inside
  the physical steering limit. **Bottom:** the scale $\\alpha_k$ (RT-MPC always uses 1).
* **The bounds** (Assumption 1) are the speed error $|\\Delta v_x|\\le\\overline{\\Delta v}$ and
  the additive disturbance $w_k\\in\\mathrm{Ell}(\\Gamma_w)$, set here by its $e_y$ semi-axis.
  As in the paper, the simulated vehicle drives at the worst case $v_{x0}+\\overline{\\Delta v}$.
* Larger bounds give a larger $\\Omega$, more tightening and less steering left to the
  nominal input. In the single lane change at $\\overline{\\Delta v}=1.25$ km/h, at 45 or
  80 km/h, or at $\\bar w_{e_y}=0.060$ m — and in the slalom already at the paper's bounds —
  RT-MPC becomes infeasible while HT-MPC still drives the manoeuvre.
* With the default (paper) bounds this reproduces the benchmark exactly: closed-loop cost
  1494.3 against 2347.9, a 36.4 % reduction.
* Everything runs on open-source solvers (Clarabel, OSQP, SCS through CVXPY).
        """
    )
st.caption("Method loaded from `LatVeh_Interactive.ipynb`; the same code produces the figure "
           "of the paper in `LatVeh_Benchmark.ipynb`.")
