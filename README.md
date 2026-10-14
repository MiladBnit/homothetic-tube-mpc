# Homothetic Tube MPC with Closed-Form Scaling — Lane-Change Case Study

Companion code for

> **Homothetic Tube MPC with Closed-Form Scaling for Uncertain Linear Systems**
> Milad Banitalebi Dehkordi, Manas Mejari and Dario Piga
> *(submitted)*


| File | Purpose |
|---|---|
| `LatVeh_Benchmark.ipynb` | **Paper figure** — lateral single-track model, lane change at constant speed; produces `fig_lane_change_comparison_alt.png` |
| `LatVeh_Interactive.ipynb` | **Interactive playground** — the same method (its first code cell is a verbatim copy of the benchmark), with the manoeuvre and the uncertainty bounds as controls and the paper figure drawn as an animation |
| `Carla_recording.mp4` | **Video** — the paper's lane change performed by HT-MPC in the CARLA simulator |
| `requirements.txt` | Open-source dependencies (no commercial solver needed) |

## The controller in CARLA

The lane change of the paper, driven by HT-MPC in the CARLA simulator (Lincoln MKZ at a
constant 60 km/h, lateral control by the closed-form homothetic tube MPC, longitudinal speed
held by a PID):

https://github.com/MiladBnit/homothetic-tube-mpc/raw/main/Carla_recording.mp4

<video src="https://github.com/MiladBnit/homothetic-tube-mpc/raw/main/Carla_recording.mp4"
       controls muted playsinline width="640"></video>

If the player above does not appear, open [`Carla_recording.mp4`](Carla_recording.mp4) —
GitHub shows a player on the file page.

## Try it

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MiladBnit/homothetic-tube-mpc/blob/main/LatVeh_Interactive.ipynb)

The notebook is self-contained: open the badge, *Runtime ▸ Run all*, then use the controls.
Choose the manoeuvre (single lane change, slalom, or your own piecewise-constant reference),
the simulation length, the speed, and the two uncertainty bounds; press **Run** (~1 s) and the
animation plays by itself.

Larger bounds mean a larger $\Omega$, more tightening and less steering left to the nominal
input. The rigid scheme runs out of steering first: around $\overline{\Delta v} = 0.35$ m/s,
at 45 km/h, or above 85 km/h, **RT-MPC's QP becomes infeasible while HT-MPC still completes
the lane change**; a little further and no design exists at all.

## Run locally

```bash
pip install -r requirements.txt
jupyter lab            # open either notebook and run all cells
```

Python ≥ 3.10. All problems are modelled in [CVXPY](https://www.cvxpy.org) and solved with
open-source solvers (Clarabel, with OSQP as fallback, and SCS for the offline SDP); no MOSEK
or Gurobi licence is required.

## Reproducing the paper figure

Run `LatVeh_Benchmark.ipynb` top to bottom. All random draws are seeded, so the figure is
regenerated exactly (up to solver tolerances).

## License

MIT — see `LICENSE`.
