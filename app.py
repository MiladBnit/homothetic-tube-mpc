"""
Streamlit front-end for the interactive HT-MPC vs RT-MPC lane-change demo.

    streamlit run app.py

Every control change re-simulates both closed loops (~1 s) and rebuilds the animation.
The engine (htmpc_latveh.py) contains a verbatim copy of the benchmark controllers.
"""
import streamlit as st
from packaging.version import Version

import htmpc_latveh as H

# Streamlit >= 1.49 replaced use_container_width with width="stretch"
_FULL = ({"width": "stretch"} if Version(st.__version__) >= Version("1.49")
         else {"use_container_width": True})

P_LEVELS = [0.80, 0.90, 0.95, 0.99, 0.995, 0.999]
MENU     = ["Single lane change (paper)", "Slalom", "Custom waypoints"]

st.set_page_config(page_title="Homothetic Tube MPC - lane change", page_icon="🚗",
                   layout="wide")


@st.cache_resource(show_spinner="Offline design: RPI ellipsoid, tightening, OCPs ...")
def get_design(p_conf: float):
    return H.design(p_conf=p_conf, verbose=False)


@st.cache_data(show_spinner=False, max_entries=64)
def run_cached(p_conf, mode, T_sim, custom):
    d = get_design(p_conf)
    return H.run(d, mode, T_sim=T_sim, custom=custom)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Controls")
    mode = st.selectbox("Manoeuvre", MENU, index=0)
    custom = st.text_input("Custom waypoints  t:y; t:y; ...  (s : m)",
                           "0:-1.75; 1.2:1.75; 3.5:0.0", disabled=(mode != "Custom waypoints"))
    T_sim = st.slider("T_sim - simulation length [s]", 2.0, 15.0,
                      float(H.PRESETS[mode]["T_sim"]), 0.5, key=f"tsim_{mode}")
    p_conf = st.select_slider("confidence level p of the uncertainty bounds", P_LEVELS, 0.99,
                              format_func=lambda v: f"{v:g}")
    panels = st.checkbox("also animate α_k and δ_s", False)
    view = st.radio("view", ["paper figure (animated)", "whole track + predictions",
                             "road view (chase camera)"], index=0)
    st.divider()
    d = get_design(p_conf)
    st.caption(
        f"Design at p = {p_conf:g}:  Ω half-width in e_y = {d.s_ey:.3f} m,  "
        f"steering margin {d.u_bounds[0]-d.tight.s_u[0]:.3f} rad of {d.u_bounds[0]:.2f},  "
        f"terminal γ = {d.tight.gamma:.4f}.  "
        f"v_x0 = {d.vx0*3.6:.0f} km/h (constant), T_s = {d.Ts} s, N = {d.N}."
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Homothetic Tube MPC vs fixed-tube robust MPC - lane change")
st.markdown(
    "Both controllers use the **same** disturbance bound and the **same** RPI ellipsoid "
    "$\\Omega$. HT-MPC scales its tube online by $\\alpha_k=\\|e_k\\|_{P^{-1}}$; RT-MPC "
    "always uses the full worst-case tube ($\\alpha_k\\equiv 1$). Change anything in the "
    "sidebar and both closed loops are re-simulated; the animation starts by itself. "
    "The vehicle travels at constant speed ($v_x = 60$ km/h); the lanes differ only in lateral offset."
)

# ── Run ───────────────────────────────────────────────────────────────────────
try:
    tr_p, tr_r, y_ref, dv = run_cached(p_conf, mode, T_sim, custom)
except ValueError as e:
    st.error(f"Invalid manoeuvre: {e}")
    st.stop()

sp, sr = H.summary(tr_p), H.summary(tr_r)

# ── Metrics ───────────────────────────────────────────────────────────────────
cols = st.columns(6)
cols[0].metric("HT-MPC", sp["status"])
cols[1].metric("RT-MPC", sr["status"])
cols[2].metric("RMS e_y error [m]", f"{sp['rms_ey']:.3f}",
               delta=f"{sp['rms_ey']-sr['rms_ey']:+.3f} vs RT", delta_color="inverse")
cols[3].metric("max |δ_s| [rad]", f"{sp['max_delta']:.3f}",
               delta=f"RT-MPC {sr['max_delta']:.3f}", delta_color="off")
cols[4].metric("mean α_k (HT)", f"{sp['mean_alpha']:.3f}")
cols[5].metric("constraint violations", f"{sp['state_violations']} / {sr['state_violations']}",
               help="HT-MPC / RT-MPC, true state outside X")

# ── Animation ─────────────────────────────────────────────────────────────────
ttl = f"{mode},  p = {p_conf:g}"
frame_ms = int(round(1000*d.Ts))
if view.startswith("paper"):
    fig = H.animate_overlay(d, tr_p, tr_r, y_ref, title=ttl, frame_ms=frame_ms, panels=panels)
elif view.startswith("whole"):
    fig = H.animate_track(d, tr_p, tr_r, y_ref, title=ttl, frame_ms=frame_ms, height=760)
else:
    fig = H.animate_road(d, tr_p, tr_r, y_ref, title=ttl, frame_ms=frame_ms, height=760)
st.plotly_chart(fig, theme=None, config={"displaylogo": False}, **_FULL)

with st.expander("What am I looking at?"):
    st.markdown(
        """
* **Paper figure (animated)** - the lane-change figure of the paper drawn progressively: the
  two trajectories $e_y(x)$ with their tube bands, $\\alpha_k\\Omega$ around HT-MPC (thin, because
  the true error $e_k = x_k - z_k$ stays small) and the full $\\Omega$ around RT-MPC. With
  *also animate α_k and δ_s* two synchronised panels show the tube scale and the steering angle.
* **confidence p** - both controllers are designed from Gaussian confidence regions at level
  $p$ for the parametric ($\\Delta v_x$) and additive uncertainty. Higher $p$ means a larger
  $\\Omega$, more constraint tightening and less steering authority for the nominal MPC:
  at $p = 0.999$ the steering margin is only 0.01 rad and even the paper manoeuvre becomes
  infeasible for both controllers; at $p = 0.9$ everything is easy.
* Because only the first predicted input is applied and its constraint is tightened by
  $\\alpha_k s_u$ instead of $s_u$, HT-MPC can use much more steering authority - hence the
  faster lane change. In the slalom, RT-MPC's OCP becomes infeasible while HT-MPC keeps going;
  the trajectory then stops and the title reports the step.
* **Whole track + predictions** shows, at every step, the nominal trajectory $z_{j|k}$ the OCP
  has just optimised and its tube ahead of the vehicle; **road view** follows each vehicle with
  a chase camera. Everything runs on open-source solvers (OSQP / Clarabel / SCS via CVXPY).
        """
    )
