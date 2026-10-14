# Homothetic Tube MPC with Closed-Form Scaling — Lane-Change Case Study

Companion code for

> **Homothetic Tube MPC with Closed-Form Scaling for Uncertain Linear Systems**
> Milad Banitalebi Dehkordi, Manas Mejari and Dario Piga
> *(submitted)*

**HT-MPC** (proposed, homothetic tube) and **RT-MPC** (rigid tube) solve the *same* QP, with
the same RPI ellipsoid $\Omega$, the same terminal cost and the same polytopic terminal set.
The only difference is the constraint tightening: HT-MPC uses the closed-form scales
$\alpha_{j|k}$ driven by the measured $\alpha_k = \|e_k\|_{P^{\star-1}}$, RT-MPC uses
$\alpha_{j|k} \equiv 1$. The uncertainty is given by **bounds** (Assumption 1): a speed-error
bound $|\Delta v_x| \le \overline{\Delta v}$ and an additive disturbance $w_k$ in an
ellipsoid.

## Contents

| File | Purpose |
|---|---|
| `LatVeh_Benchmark.ipynb` | **Paper figure** — lateral single-track model, lane change at constant speed; produces `fig_lane_change_comparison_alt.png` |
| `LatVeh_Interactive.ipynb` | **Interactive playground** — the same method (its first code cell is a verbatim copy of the benchmark), with the manoeuvre and the uncertainty bounds as controls and the paper figure drawn as an animation |
| `requirements.txt` | Open-source dependencies (no commercial solver needed) |

With its default settings the interactive notebook reproduces the benchmark run exactly
(closed-loop cost 1481.0 for HT-MPC vs 2238.1 for RT-MPC, a 33.8 % reduction).

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
