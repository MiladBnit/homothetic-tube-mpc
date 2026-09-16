# Homothetic Tube MPC — Case Studies

Companion code for the paper

> **[Paper title]** — [Authors], *[Journal]*, [year]. [DOI / arXiv link]

Two benchmark case studies compare the proposed **homothetic tube MPC (HT-MPC)** against
classical **fixed-tube robust MPC (RT-MPC)** under an identical disturbance bound and an
identical RPI ellipsoid; the only difference between the controllers is the online scale
adaptation $\alpha_k = \|e_k\|_{P^{-1}}$.

## 🎮 Try it live

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://<YOUR-APP>.streamlit.app)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MiladBnit/homothetic-tube-mpc/blob/main/LatVeh_Interactive.ipynb)

Design your own lane-change manoeuvre (single / double lane change, slalom, custom
waypoints, step or ramped reference) and watch both controllers drive it step by step —
vehicle, current tube cross-section, and the prediction the MPC is making at that instant.

## Contents

| File | Purpose |
|---|---|
| `LatVeh_Benchmark.ipynb` | **Paper figure** — lateral vehicle dynamics, lane change (`fig_lane_change_comparison_alt.png`) |
| `CSTR_Benchmark.ipynb` | **Paper figure** — continuous stirred-tank reactor (`fig_cstr.png`) |
| `htmpc_latveh.py` | Engine for the interactive demos. Section 1 is a **verbatim copy** of the benchmark's model / uncertainty / RPI / tightening / OCP code; Section 2 adds manoeuvre generators, prediction recording and the animation |
| `LatVeh_Interactive.ipynb` | Interactive notebook (Jupyter / Colab): sliders + animated road view (chase camera, vehicles to scale) |
| `app.py` | Streamlit web app: same engine, sidebar controls, live animation |
| `requirements.txt` | Open-source dependencies (no commercial solver needed) |

The two `*_Benchmark` notebooks are the reference implementations used for the paper and
are kept frozen.

## Run locally

```bash
pip install -r requirements.txt

# web app
streamlit run app.py

# or the notebook
jupyter lab LatVeh_Interactive.ipynb
```

Python ≥ 3.10. Every parameter change re-simulates both closed loops (50–150 QPs each)
in about a second.

## Deploy the web app (free)

* **Streamlit Community Cloud** — sign in with GitHub at <https://share.streamlit.io>,
  *New app* → this repository → main file `app.py` → Deploy. The app sleeps when idle and
  wakes in ~30 s on the next visit.
* **Hugging Face Spaces** — create a Space with the *Streamlit* SDK and push
  `app.py`, `htmpc_latveh.py`, `requirements.txt` and `.streamlit/config.toml`.

## Reproducing the paper figures

Run `LatVeh_Benchmark.ipynb` and `CSTR_Benchmark.ipynb` top to bottom. All random draws
are seeded, so the figures are regenerated exactly (up to solver tolerances).

## Solvers

All optimisation problems are modelled in [CVXPY](https://www.cvxpy.org) and solved with
open-source solvers: OSQP / Clarabel for the MPC quadratic programs and SCS for the small
offline SDPs. No MOSEK or Gurobi licence is needed.

## Citation

```bibtex
@article{...,
  title   = {...},
  author  = {...},
  journal = {...},
  year    = {...}
}
```

## License

MIT — see `LICENSE`.
