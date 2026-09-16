"""
Streamlit front-end for the interactive HT-MPC vs RT-MPC lane-change demo.

    streamlit run app.py

Every control change re-simulates both closed loops (~1 s) and rebuilds the animation.
The engine (htmpc_latveh.py) contains a verbatim copy of the benchmark controllers.
"""
import numpy as np
import streamlit as st
from packaging.version import Version

import htmpc_latveh as H

# Streamlit >= 1.49 replaced use_container_width with width="stretch"
_FULL = ({"width": "stretch"} if Version(st.__version__) >= Version("1.49")
         else {"use_container_width": True})

st.set_page_config(page_title="Homothetic Tube MPC - lane change", page_icon="🚗",
                   layout="wide")


@st.cache_resource(show_spinner="Offline design: RPI ellipsoid, tightening, OCPs ...")
def get_design():
    return H.design(verbose=False)


@st.cache_data(show_spinner=False, max_entries=64)
def run_cached(mode, t_lc, t_hold, y_lane, T_sim, ramp, custom, plant_seed, noise_seed):
    d = get_design()
    return H.run(d, mode, t_lc, t_hold, y_lane, T_sim, ramp, custom, plant_seed, noise_seed)


d = get_design()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Manoeuvre")
    mode = st.selectbox("Type", H.MANOEUVRES, index=0)
    is_custom = mode == H.MANOEUVRES[3]
    custom = st.text_input("Waypoints  t:y; t:y; ...  (s : m)", "0:-1.75; 1.2:1.75; 3.5:0.0",
                           disabled=not is_custom)
    t_lc   = st.slider("t_lc - first lane change [s]", 0.2, 4.0, 1.2, 0.1, disabled=is_custom)
    t_hold = st.slider("t_hold - time before switching back [s]", 0.3, 5.0, 2.0, 0.1,
                       disabled=is_custom or mode == H.MANOEUVRES[0])
    y_lane = st.slider("y_lane - lanes at ±y_lane [m]", 0.25, float(np.floor(d.y_ref_max*20)/20),
                       1.75, 0.05, disabled=is_custom)
    ramp   = st.slider("ramp - linear transition [s]  (0 = step, as in the paper)",
                       0.0, 2.0, 0.0, 0.1)
    T_sim  = st.slider("T_sim - simulation length [s]", 2.0, 15.0, 5.0, 0.5)
    st.divider()
    st.subheader("Realisation")
    c1, c2 = st.columns(2)
    plant_seed = c1.number_input("plant seed (Δv_x)", 0, 9999, 42, step=1)
    noise_seed = c2.number_input("noise seed (w_k)", 0, 9999, 7, step=1)
    speed = st.select_slider("playback speed", [0.5, 1.0, 2.0, 4.0], 1.0,
                             format_func=lambda v: f"{v:g}x")
    show_static = st.checkbox("show static summary figure", False)
    st.divider()
    st.caption(
        f"v_x0 = {d.vx0*3.6:.0f} km/h, T_s = {d.Ts} s, N = {d.N}, "
        f"Ω half-width in e_y = {d.s_ey:.3f} m, c_S = {d.rpi.c_S:.6f}. "
        f"Reference limited to |e_y| ≤ {d.y_ref_max:.2f} m."
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Homothetic Tube MPC vs fixed-tube robust MPC - lane change")
st.markdown(
    "Both controllers use the **same** disturbance bound and the **same** RPI ellipsoid "
    "$\\Omega$. HT-MPC scales its tube online by $\\alpha_k=\\|e_k\\|_{P^{-1}}$; RT-MPC "
    "always uses the full worst-case tube ($\\alpha_k\\equiv 1$). Change anything in the "
    "sidebar and both closed loops are re-simulated; press **▶ Play** on the figure."
)

# ── Run ───────────────────────────────────────────────────────────────────────
try:
    tr_p, tr_r, y_ref, dv = run_cached(mode, t_lc, t_hold, y_lane, T_sim, ramp, custom,
                                       int(plant_seed), int(noise_seed))
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
st.caption(f"True plant: Δv_x = {dv:+.4f} m/s ({dv*3.6:+.3f} km/h)")

# ── Animation ─────────────────────────────────────────────────────────────────
fig = H.animate(d, tr_p, tr_r, y_ref, title=f"{mode}, ramp {ramp:.1f} s",
                frame_ms=int(round(1000*d.Ts/speed)), height=780)
st.plotly_chart(fig, theme=None, config={"displaylogo": False}, **_FULL)

if show_static:
    st.pyplot(H.plot_static(d, tr_p, tr_r, y_ref, title=None, show=False))

with st.expander("What am I looking at?"):
    st.markdown(
        """
* **Top / middle panel** - lateral deviation $e_y$ against distance travelled. Solid line: true
  trajectory so far; triangle: the vehicle (rotated by its heading error $e_\\psi$); thick bar:
  the *current* tube cross-section ($\\alpha_k\\Omega$ for HT-MPC, $\\Omega$ for RT-MPC); dotted
  line + shaded band ahead: the nominal trajectory $z_{j|k}$ the OCP has just optimised and the
  tube $\\alpha_{j|k}\\Omega$ it tightens the constraints with.
* **Bottom panel** - the tube scale $\\alpha_k$. HT-MPC measures it from the actual error
  $e_k = x_k - z_k$; RT-MPC assumes the worst case $\\alpha_k = 1$ at every step.
* Because only the first predicted input is applied and its constraint is tightened by
  $\\alpha_k s_u$ instead of $s_u$, HT-MPC can use much more steering authority - hence the
  faster lane change. Push the manoeuvre (slalom, short hold) and RT-MPC's OCP becomes
  infeasible while HT-MPC keeps going; the panel then freezes and the title reports the step.
* Everything runs on open-source solvers (OSQP / Clarabel / SCS via CVXPY).
        """
    )
