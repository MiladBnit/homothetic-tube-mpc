"""
htmpc_latveh.py - engine for the interactive HT-MPC vs RT-MPC lane-change demo
================================================================================

Section 1 (down to the "INTERACTIVE EXTENSIONS" banner) is a VERBATIM copy of the
model / uncertainty / RPI / tightening / OCP / simulator code of
``LatVeh_Benchmark.ipynb`` (the frozen, reviewer-facing notebook).  Nothing in
that section has been modified.

Section 2 adds what the interactive demo needs on top of it:
  * ``design()``           - the offline design of the paper, packaged in a dataclass
  * manoeuvre generators   - single / double lane change, slalom, custom waypoints
  * ``simulate()``         - the same closed loop, additionally recording the
                             predicted nominal trajectory z_{j|k} and the closed-form
                             scale sequence alpha_{j|k} at every step
  * ``animate_road()``     - Plotly road-view animation: chase camera, vehicles to
                             scale, tube patch, prediction ahead (play / pause / scrub)
  * ``animate_track()``    - whole-track view of the same closed loops (fixed camera)
  * ``plot_static()``      - matplotlib summary figure
  * ``run()``              - one-call convenience wrapper

Used by ``LatVeh_Interactive.ipynb`` (Jupyter / Colab) and ``app.py`` (Streamlit).
"""

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 - VERBATIM COPY OF LatVeh_Benchmark.ipynb
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import scipy.linalg  as la
import scipy.signal  as sig
import scipy.stats   as stats
import cvxpy         as cp
import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Patch
from matplotlib.lines   import Line2D
from dataclasses import dataclass
from itertools   import product as iprod

# ── matplotlib style (IEEE single-column) ─────────────────────────────────────
FP      = 8
ONE_COL = 3.5

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":         FP,    "axes.labelsize":   FP,
    "axes.titlesize":    FP,    "xtick.labelsize":  FP - 1,
    "ytick.labelsize":   FP - 1,"legend.fontsize":  FP - 1,
    "legend.framealpha": 0.90,  "legend.frameon":   True,
    "legend.edgecolor":  "0.80","lines.linewidth":  1.0,
    "axes.linewidth":    0.6,   "grid.linewidth":   0.4,
    "grid.alpha":        0.40,  "savefig.dpi":      300,
    "savefig.bbox":      "tight","pdf.fonttype":    42,
    "ps.fonttype":       42,
})

C = {
    "p":      "#007E5C",     # HT-MPC  — green
    "r":      "#FF4500",     # RT-MPC  — orange-red
    "ref":    "#1D1D1DCA",   # reference — near-black
    "constr": "#A80000",     # lane boundaries — dark red
}
LABELS = {"p": "HT-MPC", "r": "RT-MPC"}


# ══════════════════════════════════════════════════════════════════════════════
#  VEHICLE MODEL
# ══════════════════════════════════════════════════════════════════════════════

_m  = 1500.0;  _Iz = 2500.0
_lf = 1.2;     _lr = 1.6
_Cf = 60000.0; _Cr = 70000.0


def _Ac(vx):
    m, Iz, lf, lr, Cf, Cr = _m, _Iz, _lf, _lr, _Cf, _Cr
    return np.array([
        [0.,  vx,  1.,  0.],
        [0.,  0.,  0.,  1.],
        [0.,  0.,  -2.*(Cf+Cr)/(m*vx),
                   (-2.*(Cf*lf-Cr*lr)/(m*vx) - vx)],
        [0.,  0.,  -2.*(Cf*lf-Cr*lr)/(Iz*vx),
                   -2.*(Cf*lf**2+Cr*lr**2)/(Iz*vx)],
    ])


def _Bc():
    m, Iz, lf, Cf = _m, _Iz, _lf, _Cf
    return np.array([
        [0.,           0.   ],
        [0.,           0.   ],
        [2.*Cf/m,      0.   ],
        [2.*Cf*lf/Iz,  1./Iz],
    ])


def discretize(vx, Ts):
    sd = sig.StateSpace(
        _Ac(vx), _Bc(), np.eye(4), np.zeros((4, 2))
    ).to_discrete(Ts)
    return np.array(sd.A), np.array(sd.B)


def dAd_dvx(vx0, Ts, h=1e-4):
    """Numerical Jacobian D = dA_d/dv_x at v_x0."""
    Ap, _ = discretize(vx0 + h, Ts)
    Am, _ = discretize(vx0 - h, Ts)
    return (Ap - Am) / (2.*h)


# ══════════════════════════════════════════════════════════════════════════════
#  UNCERTAINTY MODEL  (shared by both controllers)
# ══════════════════════════════════════════════════════════════════════════════

class UncertaintyModel:
    """
    Gaussian confidence-region instantiation of the bounded sets A, B, W.
    df_DA = 1 because Delta_A = Delta_vx * D is driven by ONE scalar.
    """
    def __init__(self, Sigma_DA, Sigma_DB, Sigma_w,
                 p_A, p_B, p_W, n, m, df_DA=1, df_DB=1, label=""):
        self.Sigma_DA = np.atleast_2d(Sigma_DA)
        self.Sigma_DB = np.atleast_2d(Sigma_DB)
        self.Sigma_w  = np.atleast_2d(Sigma_w)
        self.p_A = p_A;  self.p_B = p_B;  self.p_W = p_W
        self.n = n;  self.m = m
        self.chi2_DA = float(stats.chi2.ppf(p_A, df=df_DA))
        self.chi2_DB = float(stats.chi2.ppf(p_B, df=df_DB))
        self.chi2_w  = float(stats.chi2.ppf(p_W, df=n))
        tag = f" [{label}]" if label else ""
        print(f"  OK{tag}  p_A={p_A}, p_B={p_B}, p_W={p_W}  "
              f"chi2_DA={self.chi2_DA:.3f}  chi2_w={self.chi2_w:.3f}")

    def QA_at_x(self, x):
        L = np.kron(np.eye(self.n), x.reshape(1, -1))
        Q = L @ self.Sigma_DA @ L.T
        return 0.5*(Q + Q.T)

    def QB_at_u(self, u):
        L = np.kron(np.eye(self.n), u.reshape(1, -1))
        Q = L @ self.Sigma_DB @ L.T
        return 0.5*(Q + Q.T)


# ══════════════════════════════════════════════════════════════════════════════
#  VERTEX BOUND (Lemma 1 of the paper)
# ══════════════════════════════════════════════════════════════════════════════

def _box_vertices(H, h):
    d    = H.shape[1];  half = len(h)//2
    assert H.shape[0] == 2*d
    assert np.allclose(H, np.vstack([np.eye(d), -np.eye(d)]))
    return np.array(list(iprod(*zip(-h[half:], h[:d]))))


def compute_Qmax(unc, Hx, h_x, Hu, h_u):
    n = unc.n
    def _sdp(mats):
        Qv = cp.Variable((n, n), symmetric=True)
        cp.Problem(
            cp.Minimize(cp.trace(Qv)),
            [Qv >> M + 1e-9*np.eye(n) for M in mats]
        ).solve(solver="SCS", verbose=False, eps=1e-8, max_iters=15_000)
        if Qv.value is None:
            return mats[0] + 1e-6*np.eye(n)
        return 0.5*(Qv.value + Qv.value.T) + 1e-9*np.eye(n)
    QA_max = _sdp([unc.QA_at_x(v) for v in _box_vertices(Hx, h_x)])
    QB_max = _sdp([unc.QB_at_u(v) for v in _box_vertices(Hu, h_u)])
    print(f"  OK  tr(Q^A_max)={np.trace(QA_max):.5f}  "
          f"tr(Q^B_max)={np.trace(QB_max):.5f}")
    return QA_max, QB_max


# ══════════════════════════════════════════════════════════════════════════════
#  RPI ELLIPSOID  (Prop. 1 of the paper — Stein closed form)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RPIData:
    P_star:     np.ndarray
    P_star_inv: np.ndarray
    c_S:        float
    lam_star:   float
    S_max:      np.ndarray


def compute_rpi(A_K, unc, QA_max, QB_max, label=""):
    a = [unc.chi2_DA*float(np.trace(QA_max)),
         unc.chi2_DB*float(np.trace(QB_max)),
         unc.chi2_w *float(np.trace(unc.Sigma_w))]
    s = sum(np.sqrt(ai) for ai in a)
    mu = np.array([np.sqrt(ai)/s for ai in a])
    S_max = ((unc.chi2_DA/mu[0])*QA_max
           + (unc.chi2_DB/mu[1])*QB_max
           + (unc.chi2_w /mu[2])*unc.Sigma_w)
    S_max = 0.5*(S_max + S_max.T)

    def _stein(lam):
        P = la.solve_discrete_lyapunov(A_K/np.sqrt(lam), S_max/(1.-lam))
        return 0.5*(P + P.T)

    rho2 = float(np.max(np.abs(la.eigvals(A_K))))**2
    a_, b_ = rho2+1e-8, 1.-1e-8
    gr = (np.sqrt(5.)-1.)/2.
    x1 = b_-gr*(b_-a_);  f1 = np.trace(_stein(x1))
    x2 = a_+gr*(b_-a_);  f2 = np.trace(_stein(x2))
    for _ in range(400):
        if b_-a_ < 1e-12: break
        if f1 < f2:
            b_,x2,f2 = x2,x1,f1; x1=b_-gr*(b_-a_); f1=np.trace(_stein(x1))
        else:
            a_,x1,f1 = x1,x2,f2; x2=a_+gr*(b_-a_); f2=np.trace(_stein(x2))
    lam_star = 0.5*(a_+b_)
    P_star   = _stein(lam_star)
    Pnh      = np.real(la.inv(la.sqrtm(P_star)))
    raw_cS   = (float(np.max(np.real(la.eigvals(Pnh@S_max@Pnh))))
                /(1.-lam_star))
    c_S = float(np.clip(raw_cS, 1e-12, 1.))
    tag = f" [{label}]" if label else ""
    print(f"  OK{tag}  lam*={lam_star:.4f}  c_S={c_S:.6f}  "
          f"tr(P*)={np.trace(P_star):.4f}  "
          f"sqrt(P*[0,0])={np.sqrt(P_star[0,0]):.4f} m")
    return RPIData(P_star=P_star, P_star_inv=la.inv(P_star),
                   c_S=c_S, lam_star=lam_star, S_max=S_max)


# ══════════════════════════════════════════════════════════════════════════════
#  TIGHTENING + TERMINAL SET
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TightData:
    s_x:    np.ndarray
    s_u:    np.ndarray
    gamma:  float
    L_inf:  object
    sq_gam: float


def compute_tightening(A_K, K, rpi, Hx, h_x, Hu, h_u, P_inf, label=""):
    P = rpi.P_star;  Pi = la.inv(P_inf);  KPK = K@P@K.T
    s_x = np.array([np.sqrt(max(float(h@P@h),   0.)) for h in Hx])
    s_u = np.array([np.sqrt(max(float(h@KPK@h), 0.)) for h in Hu])
    gammas = []
    for i, h in enumerate(Hx):
        mg = h_x[i]-s_x[i];  sv = float(h@Pi@h)
        if mg > 1e-10 and sv > 1e-12: gammas.append(mg**2/sv)
    for i, h in enumerate(Hu):
        hz = K.T@h;  mg = h_u[i]-s_u[i];  sv = float(hz@Pi@hz)
        if mg > 1e-10 and sv > 1e-12: gammas.append(mg**2/sv)
    gamma  = float(np.min(gammas)) if gammas else 0.
    L_inf  = np.linalg.cholesky(P_inf).T if gamma > 1e-8 else None
    sq_gam = float(np.sqrt(gamma))       if gamma > 1e-8 else 0.
    tag = f" [{label}]" if label else ""
    print(f"  OK{tag}  s_x={np.round(s_x,4)}  gamma={gamma:.5f}  "
          f"margins: {np.round(h_x-s_x,4)}")
    return TightData(s_x=s_x, s_u=s_u, gamma=gamma,
                     L_inf=L_inf, sq_gam=sq_gam)


# ══════════════════════════════════════════════════════════════════════════════
#  CLOSED-FORM ALPHA SEQUENCE  (Prop. 3 of the paper)
# ══════════════════════════════════════════════════════════════════════════════

def alpha_seq(alpha_k, c_S, N):
    j = np.arange(N+1, dtype=float)
    return np.sqrt(np.maximum(1.-(1.-c_S)**j*(1.-alpha_k**2), 0.))


# ══════════════════════════════════════════════════════════════════════════════
#  STEADY STATE
# ══════════════════════════════════════════════════════════════════════════════

def steady_state(A, B, C_out, r_ref):
    n_, m_ = A.shape[0], B.shape[1];  p_ = C_out.shape[0]
    r = np.atleast_1d(float(r_ref))
    M   = np.block([[A-np.eye(n_), B], [C_out, np.zeros((p_,m_))]])
    rhs = np.concatenate([np.zeros(n_), r])
    sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    return sol[:n_], sol[n_:]


# ══════════════════════════════════════════════════════════════════════════════
#  HT-MPC OCP
# ══════════════════════════════════════════════════════════════════════════════

class HomotheticTubeMPC_OCP:
    """
    Proposed method: per-step tightening driven by the precomputed
    closed-form scale sequence {alpha_{j|k}}.  Standard QP, no SOC
    on the tube scale (only on the terminal set).
    """
    def __init__(self, A, B, Hx, h_x, Hu, h_u, tight, N, Q_c, R_c, P_inf):
        print("[Compile] HT-MPC OCP ...")
        self.A=A; self.B=B; self.Hx=Hx; self.h_x=h_x
        self.Hu=Hu; self.h_u=h_u; self.tight=tight; self.N=N
        n=A.shape[0]; m=B.shape[1]
        Q_c=np.atleast_2d(Q_c); R_c=np.atleast_2d(R_c)
        use_t = tight.gamma > 1e-8
        self.z0_p=cp.Parameter(n); self.zr_p=cp.Parameter(n)
        self.vr_p=cp.Parameter(m)
        self.hz_p=[cp.Parameter(Hx.shape[0]) for _ in range(N+1)]
        self.hv_p=[cp.Parameter(Hu.shape[0]) for _ in range(N)]
        z=cp.Variable((N+1,n)); v=cp.Variable((N,m))
        self._z=z; self._v=v
        cost=0.; cons=[z[0]==self.z0_p]
        for j in range(N):
            cost+=(cp.quad_form(z[j]-self.zr_p,Q_c)
                  +cp.quad_form(v[j]-self.vr_p,R_c))
            cons+=[z[j+1]==A@z[j]+B@v[j],
                   Hx@z[j]<=self.hz_p[j],
                   Hu@v[j]<=self.hv_p[j]]
        cost+=cp.quad_form(z[N]-self.zr_p,P_inf)
        cons.append(Hx@z[N]<=self.hz_p[N])
        if use_t:
            cons.append(cp.norm(tight.L_inf@(z[N]-self.zr_p),2)<=tight.sq_gam)
        self.prob=cp.Problem(cp.Minimize(cost),cons)
        self._solver=cp.CLARABEL if use_t else cp.OSQP
        print(f"  OK  N={N}  solver={self._solver}  terminal_SOC={use_t}")

    def solve(self, z_k, z_ref, v_ref, aseq, warm=True):
        self.z0_p.value=z_k; self.zr_p.value=z_ref; self.vr_p.value=v_ref
        for j in range(self.N+1):
            self.hz_p[j].value=self.h_x-aseq[j]*self.tight.s_x
        for j in range(self.N):
            self.hv_p[j].value=self.h_u-aseq[j]*self.tight.s_u
        try:
            kw = dict(warm_start=warm, verbose=False)
            if self._solver==cp.OSQP:
                self.prob.solve(solver=cp.OSQP,eps_abs=1e-7,eps_rel=1e-7,
                                max_iter=40_000,polish=True,**kw)
            else:
                self.prob.solve(solver=cp.CLARABEL,**kw)
            ok=self.prob.status in ("optimal","optimal_inaccurate")
            return (self._z.value.copy(),self._v.value.copy(),True) if ok \
                   else (None,None,False)
        except Exception as e:
            print(f"  HT-MPC OCP error: {e}"); return None,None,False


# ══════════════════════════════════════════════════════════════════════════════
#  RT-MPC OCP  (fixed tube, alpha = 1 always)
# ══════════════════════════════════════════════════════════════════════════════

class FixedTubeMPC_OCP:
    """
    Classical robust tube MPC.  alpha = 1 always -> fixed worst-case
    tightening.  Uses the SAME Omega (and hence same s_x, s_u) as HT-MPC.
    """
    def __init__(self, A, B, Hx, h_x, Hu, h_u, tight, N, Q_c, R_c, P_inf):
        print("[Compile] RT-MPC OCP ...")
        self.N=N; n=A.shape[0]; m=B.shape[1]
        Q_c=np.atleast_2d(Q_c); R_c=np.atleast_2d(R_c)
        use_t=tight.gamma>1e-8
        hz_fix=h_x-tight.s_x; hv_fix=h_u-tight.s_u
        self.z0_p=cp.Parameter(n); self.zr_p=cp.Parameter(n)
        self.vr_p=cp.Parameter(m)
        z=cp.Variable((N+1,n)); v=cp.Variable((N,m))
        self._z=z; self._v=v
        cost=0.; cons=[z[0]==self.z0_p]
        for j in range(N):
            cost+=(cp.quad_form(z[j]-self.zr_p,Q_c)
                  +cp.quad_form(v[j]-self.vr_p,R_c))
            cons+=[z[j+1]==A@z[j]+B@v[j],
                   Hx@z[j]<=hz_fix,
                   Hu@v[j]<=hv_fix]
        cost+=cp.quad_form(z[N]-self.zr_p,P_inf)
        cons.append(Hx@z[N]<=hz_fix)
        if use_t:
            cons.append(cp.norm(tight.L_inf@(z[N]-self.zr_p),2)<=tight.sq_gam)
        self.prob=cp.Problem(cp.Minimize(cost),cons)
        self._solver=cp.CLARABEL if use_t else cp.OSQP
        print(f"  OK  fixed tightening h_x-s_x={np.round(hz_fix[:4],4)}")

    def solve(self, z_k, z_ref, v_ref, warm=True):
        self.z0_p.value=z_k; self.zr_p.value=z_ref; self.vr_p.value=v_ref
        try:
            kw=dict(warm_start=warm,verbose=False)
            if self._solver==cp.OSQP:
                self.prob.solve(solver=cp.OSQP,eps_abs=1e-7,eps_rel=1e-7,
                                max_iter=40_000,polish=True,**kw)
            else:
                self.prob.solve(solver=cp.CLARABEL,**kw)
            ok=self.prob.status in ("optimal","optimal_inaccurate")
            return (self._z.value.copy(),self._v.value.copy(),True) if ok \
                   else (None,None,False)
        except Exception as e:
            print(f"  RT-MPC OCP error: {e}"); return None,None,False


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimResult:
    x:      np.ndarray
    z:      np.ndarray
    u:      np.ndarray
    alpha:  np.ndarray
    y_ref:  np.ndarray
    viol_x: np.ndarray
    viol_u: np.ndarray
    n_viol: int
    ok:     bool


def run_htmpc(A, B, A_hat, B_hat, K, C_out,
              Hx, h_x, Hu, h_u, Sigma_w,
              rpi, ocp, y_ref_sig, x0, N, rng_seed=None):
    rng=np.random.RandomState(rng_seed)
    T=len(y_ref_sig); n=A.shape[0]; m=B.shape[1]
    Xt=np.zeros((T,n)); Zt=np.zeros((T,n)); Ut=np.zeros((T,m))
    At=np.zeros(T); vx=np.zeros(T,bool); vu=np.zeros(T,bool)
    ok=True; n_viol=0
    x_k=x0.copy(); z_k=x0.copy()
    for k in range(T):
        zr_k,vr_k=steady_state(A,B,C_out,float(y_ref_sig[k]))
        e_k=x_k-z_k
        raw=float(np.sqrt(max(float(e_k@rpi.P_star_inv@e_k),0.)))
        if raw>1.+1e-6: n_viol+=1
        At[k]=raw; alpha_k=min(raw,1.)
        aseq=alpha_seq(alpha_k,rpi.c_S,N)
        z_opt,v_opt,feas=ocp.solve(z_k,zr_k,vr_k,aseq)
        if not feas: ok=False; break
        u_k=v_opt[0]+K@e_k
        Xt[k]=x_k; Zt[k]=z_k; Ut[k]=u_k
        vx[k]=bool(np.any(Hx@x_k>h_x+1e-6))
        vu[k]=bool(np.any(Hu@u_k>h_u+1e-6))
        w_k=rng.multivariate_normal(np.zeros(n),Sigma_w)
        x_k=A_hat@x_k+B_hat@u_k+w_k
        z_k=A@Zt[k]+B@v_opt[0]
    Te=T if ok else k
    return SimResult(x=Xt[:Te],z=Zt[:Te],u=Ut[:Te],alpha=At[:Te],
                     y_ref=y_ref_sig[:Te],viol_x=vx[:Te],viol_u=vu[:Te],
                     n_viol=n_viol,ok=ok)


def run_rtmpc(A, B, A_hat, B_hat, K, C_out,
              Hx, h_x, Hu, h_u, Sigma_w,
              rpi, ocp, y_ref_sig, x0, N, rng_seed=None):
    rng=np.random.RandomState(rng_seed)
    T=len(y_ref_sig); n=A.shape[0]; m=B.shape[1]
    Xt=np.zeros((T,n)); Zt=np.zeros((T,n)); Ut=np.zeros((T,m))
    At=np.ones(T); vx=np.zeros(T,bool); vu=np.zeros(T,bool)
    ok=True
    x_k=x0.copy(); z_k=x0.copy()
    for k in range(T):
        zr_k,vr_k=steady_state(A,B,C_out,float(y_ref_sig[k]))
        e_k=x_k-z_k
        z_opt,v_opt,feas=ocp.solve(z_k,zr_k,vr_k)
        if not feas: ok=False; break
        u_k=v_opt[0]+K@e_k
        Xt[k]=x_k; Zt[k]=z_k; Ut[k]=u_k
        vx[k]=bool(np.any(Hx@x_k>h_x+1e-6))
        vu[k]=bool(np.any(Hu@u_k>h_u+1e-6))
        w_k=rng.multivariate_normal(np.zeros(n),Sigma_w)
        x_k=A_hat@x_k+B_hat@u_k+w_k
        z_k=A@Zt[k]+B@v_opt[0]
    Te=T if ok else k
    return SimResult(x=Xt[:Te],z=Zt[:Te],u=Ut[:Te],alpha=At[:Te],
                     y_ref=y_ref_sig[:Te],viol_x=vx[:Te],viol_u=vu[:Te],
                     n_viol=0,ok=ok)




# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 - INTERACTIVE EXTENSIONS  (not part of the benchmark)
# ══════════════════════════════════════════════════════════════════════════════

import io
import contextlib
from typing import List, Optional, Tuple


# ── Offline design (identical numbers to the paper) ───────────────────────────

@dataclass
class Design:
    A: np.ndarray;  B: np.ndarray;  K: np.ndarray;  C_out: np.ndarray
    Hx: np.ndarray; h_x: np.ndarray; Hu: np.ndarray; h_u: np.ndarray
    x_bounds: np.ndarray; u_bounds: np.ndarray
    Sigma_w: np.ndarray
    rpi: RPIData; tight: TightData
    ocp_p: HomotheticTubeMPC_OCP; ocp_r: FixedTubeMPC_OCP
    N: int; Ts: float; vx0: float; sigma_vx: float
    lane_width: float; y_ref_max: float
    log: str = ""

    @property
    def s_ey(self) -> float:
        """Half-width of the RPI ellipsoid Omega in the e_y direction."""
        return float(np.sqrt(self.rpi.P_star[0, 0]))


def design(verbose: bool = True) -> Design:
    """Offline design of LatVeh_Benchmark.ipynb (same parameters, same numbers)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        vx0, Ts = 60.0/3.6, 0.1
        A, B  = discretize(vx0, Ts)
        n, m  = 4, 2
        C_out = np.array([[1., 0., 0., 0.]])

        Q_lqr = np.diag([50., 10., 1., 10.]);  R_lqr = np.diag([1.0, 0.1])
        P_lqr = la.solve_discrete_are(A, B, Q_lqr, R_lqr)
        K     = -la.solve(R_lqr + B.T@P_lqr@B, B.T@P_lqr@A)
        A_K   = A + B@K
        assert float(np.max(np.abs(la.eigvals(A_K)))) < 1.0
        assert float(abs(la.det(A_K))) > 1e-9

        sigma_vx = 0.1
        D        = dAd_dvx(vx0, Ts);  d_flat = D.flatten()
        eps_reg  = (sigma_vx*0.01)**2
        Sigma_DA = sigma_vx**2*np.outer(d_flat, d_flat) + eps_reg*np.eye(n*n)
        Sigma_DA = 0.5*(Sigma_DA + Sigma_DA.T)
        Sigma_DB = (sigma_vx*1e-3)**2*np.eye(n*m)
        Sigma_w  = np.diag([1e-4, 2e-5, 1e-4, 5e-5])*2
        p_conf   = 0.99

        x_bounds = np.array([3.0, 0.30, 2.0, 1.5]);  u_bounds = np.array([0.20, 2.0])
        Hx = np.vstack([np.eye(n), -np.eye(n)]);  h_x = np.tile(x_bounds, 2)
        Hu = np.vstack([np.eye(m), -np.eye(m)]);  h_u = np.tile(u_bounds, 2)

        Q_c = np.diag([20., 5., 0.5, 0.5]);  R_c = np.diag([0.1, 0.01])
        P_inf = la.solve_discrete_are(A, B, Q_c, R_c)

        unc    = UncertaintyModel(Sigma_DA, Sigma_DB, Sigma_w, p_conf, p_conf, p_conf,
                                  n, m, df_DA=1, df_DB=1, label="shared")
        QA, QB = compute_Qmax(unc, Hx, h_x, Hu, h_u)
        rpi    = compute_rpi(A_K, unc, QA, QB, label="shared")
        tight  = compute_tightening(A_K, K, rpi, Hx, h_x, Hu, h_u, P_inf, label="shared")

        N = 25
        ocp_p = HomotheticTubeMPC_OCP(A, B, Hx, h_x, Hu, h_u, tight, N, Q_c, R_c, P_inf)
        ocp_r = FixedTubeMPC_OCP(     A, B, Hx, h_x, Hu, h_u, tight, N, Q_c, R_c, P_inf)

    d = Design(A=A, B=B, K=K, C_out=C_out, Hx=Hx, h_x=h_x, Hu=Hu, h_u=h_u,
               x_bounds=x_bounds, u_bounds=u_bounds, Sigma_w=Sigma_w,
               rpi=rpi, tight=tight, ocp_p=ocp_p, ocp_r=ocp_r,
               N=N, Ts=Ts, vx0=vx0, sigma_vx=sigma_vx, lane_width=3.5,
               y_ref_max=float(x_bounds[0] - tight.s_x[0] - 0.25), log=buf.getvalue())
    if verbose:
        print(d.log, end="")
        print(f"  Omega half-width in e_y: {d.s_ey:.4f} m   "
              f"c_S = {rpi.c_S:.6f}   reference limit |e_y| <= {d.y_ref_max:.2f} m")
    return d


# ── Manoeuvre generators ──────────────────────────────────────────────────────

MANOEUVRES = [
    "Single lane change (paper)",
    "Double lane change (overtake & return)",
    "Slalom",
    "Custom waypoints",
]


def parse_waypoints(text: str) -> List[Tuple[float, float]]:
    """'0:-1.75; 1.2:1.75; 3:-1.75'  ->  [(0.0,-1.75), (1.2,1.75), (3.0,-1.75)]"""
    pts = []
    for tok in text.replace(",", ";").split(";"):
        tok = tok.strip()
        if not tok:
            continue
        try:
            t, y = tok.split(":")
            pts.append((float(t), float(y)))
        except ValueError:
            raise ValueError(f"cannot parse waypoint '{tok}' - expected 't:y', "
                             f"e.g. '0:-1.75; 1.2:1.75'") from None
    if not pts:
        raise ValueError("no waypoints given")
    pts.sort()
    if pts[0][0] > 0.:
        pts.insert(0, (0., pts[0][1]))
    return pts


def waypoints_for(mode, t_lc=1.2, t_hold=2.0, y_lane=1.75, T_sim=5.0, custom=""):
    y_slow, y_fast = -y_lane, +y_lane
    if mode == MANOEUVRES[0]:
        return [(0., y_slow), (t_lc, y_fast)]
    if mode == MANOEUVRES[1]:
        return [(0., y_slow), (t_lc, y_fast), (t_lc + t_hold, y_slow)]
    if mode == MANOEUVRES[2]:
        pts, t, y = [(0., y_slow)], t_lc, y_fast
        while t < T_sim:
            pts.append((t, y));  t += t_hold;  y = -y
        return pts
    if mode == MANOEUVRES[3]:
        return parse_waypoints(custom)
    raise ValueError(f"unknown manoeuvre '{mode}'")


def build_reference(waypoints, T_sim, Ts, ramp=0., y_max=None) -> np.ndarray:
    """Piecewise-constant e_y reference on the grid k*Ts, with optional linear ramps."""
    T = int(round(T_sim/Ts))
    t = np.arange(T)*Ts
    y = np.full(T, waypoints[0][1], float)
    for (t_i, y_i) in waypoints[1:]:
        y[t >= t_i - 1e-9] = y_i
    if ramp > 0.:
        n_r    = max(int(round(ramp/Ts)), 1)
        y_step = y.copy()
        for k in np.where(np.diff(y_step) != 0.)[0] + 1:
            k_end = min(k + n_r, T)
            y[k:k_end] = np.linspace(y_step[k-1], y_step[k], k_end - k + 1)[1:]
    if y_max is not None and np.max(np.abs(y)) > y_max + 1e-9:
        raise ValueError(f"|e_y| reference must stay within +/-{y_max:.2f} m "
                         f"(got {np.max(np.abs(y)):.2f} m).")
    return y


# ── Closed loop with prediction recording ────────────────────────────────────

@dataclass
class Trace:
    """Closed-loop record.  Same fields as SimResult plus the per-step predictions."""
    x:      np.ndarray   # (T, n)   true state
    z:      np.ndarray   # (T, n)   nominal state
    u:      np.ndarray   # (T, m)   applied input
    alpha:  np.ndarray   # (T,)     tube scale alpha_k (1 for RT-MPC)
    y_ref:  np.ndarray   # (T,)
    viol_x: np.ndarray   # (T,) bool
    viol_u: np.ndarray   # (T,) bool
    z_pred: np.ndarray   # (T, N+1, n)  predicted nominal trajectory at step k
    a_pred: np.ndarray   # (T, N+1)     tube scale along the horizon at step k
    ok:     bool
    T_full: int          # requested length (== len(x) unless infeasible)


def _simulate_one(d: Design, kind: str, A_hat, B_hat, y_ref_sig, x0, rng_seed) -> Trace:
    """Identical loop to run_htmpc / run_rtmpc, additionally storing z_{j|k}, alpha_{j|k}."""
    A, B, K, C_out, N = d.A, d.B, d.K, d.C_out, d.N
    Hx, h_x, Hu, h_u  = d.Hx, d.h_x, d.Hu, d.h_u
    rpi, Sigma_w      = d.rpi, d.Sigma_w
    ocp = d.ocp_p if kind == "p" else d.ocp_r
    rng = np.random.RandomState(rng_seed)
    T = len(y_ref_sig);  n = A.shape[0];  m = B.shape[1]
    Xt = np.zeros((T, n)); Zt = np.zeros((T, n)); Ut = np.zeros((T, m))
    At = np.ones(T);  vx = np.zeros(T, bool);  vu = np.zeros(T, bool)
    Zp = np.zeros((T, N+1, n));  Ap = np.ones((T, N+1))
    ok = True
    x_k = x0.copy();  z_k = x0.copy()
    for k in range(T):
        zr_k, vr_k = steady_state(A, B, C_out, float(y_ref_sig[k]))
        e_k = x_k - z_k
        if kind == "p":
            raw = float(np.sqrt(max(float(e_k@rpi.P_star_inv@e_k), 0.)))
            At[k] = raw;  alpha_k = min(raw, 1.)
            aseq = alpha_seq(alpha_k, rpi.c_S, N)
            z_opt, v_opt, feas = ocp.solve(z_k, zr_k, vr_k, aseq)
            Ap[k] = aseq
        else:
            z_opt, v_opt, feas = ocp.solve(z_k, zr_k, vr_k)
        if not feas:
            ok = False;  break
        u_k = v_opt[0] + K@e_k
        Xt[k] = x_k;  Zt[k] = z_k;  Ut[k] = u_k;  Zp[k] = z_opt
        vx[k] = bool(np.any(Hx@x_k > h_x + 1e-6))
        vu[k] = bool(np.any(Hu@u_k > h_u + 1e-6))
        w_k = rng.multivariate_normal(np.zeros(n), Sigma_w)
        x_k = A_hat@x_k + B_hat@u_k + w_k
        z_k = A@Zt[k] + B@v_opt[0]
    Te = T if ok else k
    return Trace(x=Xt[:Te], z=Zt[:Te], u=Ut[:Te], alpha=At[:Te], y_ref=y_ref_sig[:Te],
                 viol_x=vx[:Te], viol_u=vu[:Te], z_pred=Zp[:Te], a_pred=Ap[:Te],
                 ok=ok, T_full=T)


def simulate(d: Design, y_ref_sig, plant_seed=42, noise_seed=7):
    """One true plant (fixed Delta_vx draw) + one noise sequence; both controllers."""
    Delta_vx     = float(np.random.RandomState(plant_seed).normal(0., d.sigma_vx))
    A_hat, B_hat = discretize(d.vx0 + Delta_vx, d.Ts)
    x0 = np.array([y_ref_sig[0], 0., 0., 0.])
    tr_p = _simulate_one(d, "p", A_hat, B_hat, y_ref_sig, x0, noise_seed)
    tr_r = _simulate_one(d, "r", A_hat, B_hat, y_ref_sig, x0, noise_seed)
    return tr_p, tr_r, Delta_vx


def summary(tr: Trace) -> dict:
    T = len(tr.x)
    return {
        "status":           "OK" if tr.ok else f"infeasible at k = {T}",
        "ok":               tr.ok,
        "steps":            T,
        "state_violations": int(tr.viol_x.sum()),
        "input_violations": int(tr.viol_u.sum()),
        "mean_alpha":       float(tr.alpha.mean()) if T else float("nan"),
        "max_alpha":        float(tr.alpha.max())  if T else float("nan"),
        "rms_ey":           float(np.sqrt(np.mean((tr.x[:, 0]-tr.y_ref)**2))) if T else float("nan"),
        "max_delta":        float(np.max(np.abs(tr.u[:, 0]))) if T else float("nan"),
    }


def print_summary(tr_p: Trace, tr_r: Trace, Delta_vx: float):
    sp, sr = summary(tr_p), summary(tr_r)
    print(f"  True plant: Delta_vx = {Delta_vx:+.4f} m/s ({Delta_vx*3.6:+.3f} km/h)")
    print(f"  {'':20s}{'HT-MPC':>22s}{'RT-MPC':>22s}")
    rows = [("status", "status", "{}"), ("state violations", "state_violations", "{}"),
            ("input violations", "input_violations", "{}"), ("mean alpha_k", "mean_alpha", "{:.4f}"),
            ("max alpha_k", "max_alpha", "{:.4f}"), ("RMS e_y error [m]", "rms_ey", "{:.4f}"),
            ("max |delta_s| [rad]", "max_delta", "{:.4f}")]
    for lbl, key, fmt in rows:
        print(f"  {lbl:20s}{fmt.format(sp[key]):>22s}{fmt.format(sr[key]):>22s}")


# ── Plotly animation ──────────────────────────────────────────────────────────

_HEX = {"p": "#007E5C", "r": "#FF4500", "ref": "#1D1D1D", "constr": "#A80000"}


def _rgba(hexcol, a):
    h = hexcol.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"


def animate_track(d: Design, tr_p: Trace, tr_r: Trace, y_ref_sig, title=None,
                  frame_ms: int = 100, height: int = 820):
    """
    Whole-track Plotly animation of both closed loops (fixed camera).
      row 1: HT-MPC  - past path, vehicle, current tube alpha_k*Omega, predicted
                       nominal trajectory z_{j|k} with the tube alpha_{j|k}*Omega
      row 2: RT-MPC  - same with the fixed tube Omega
      row 3: alpha_k of both controllers, with a cursor
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    Ts, vx0, N, s, W = d.Ts, d.vx0, d.N, d.s_ey, d.lane_width
    T_full = len(y_ref_sig)
    T_max  = max(len(tr_p.x), len(tr_r.x))
    ds     = Ts*vx0                                   # metres per step
    x_end  = (T_full - 1 + N)*ds
    s_ref  = np.arange(T_full + N)*ds
    y_ref_ext = np.concatenate([y_ref_sig, np.full(N, y_ref_sig[-1])])

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.38, 0.38, 0.24], vertical_spacing=0.05,
                        subplot_titles=("HT-MPC (homothetic tube, proposed)",
                                        "RT-MPC (fixed tube)",
                                        "tube scale α<sub>k</sub>"))

    # static background per spatial row
    for row in (1, 2):
        for yv, wdt, dash in ((W, 1.5, "solid"), (-W, 1.5, "solid"),
                              (d.x_bounds[0], 0.8, "dash"), (-d.x_bounds[0], 0.8, "dash"),
                              (0., 0.6, "dot")):
            fig.add_shape(type="line", x0=0, x1=x_end, y0=yv, y1=yv, row=row, col=1,
                          line=dict(color=_HEX["constr"], width=wdt, dash=dash), opacity=0.7)
        fig.add_trace(go.Scatter(x=s_ref, y=y_ref_ext, mode="lines", line_shape="hv",
                                 line=dict(color=_HEX["ref"], width=1), name="reference",
                                 showlegend=(row == 1), hoverinfo="skip"), row=row, col=1)
    fig.add_shape(type="line", x0=0, x1=x_end, y0=1, y1=1, row=3, col=1,
                  line=dict(color="grey", width=0.6, dash="dot"))

    # animated traces - keep the order, the frames refer to these indices
    anim_idx = {}

    def _add(key, trace, row):
        anim_idx[key] = len(fig.data)
        fig.add_trace(trace, row=row, col=1)

    for key, tr, row in (("p", tr_p, 1), ("r", tr_r, 2)):
        col, lbl = _HEX[key], LABELS[key]
        _add(f"{key}_band", go.Scatter(x=[], y=[], mode="lines", fill="toself",
                                       fillcolor=_rgba(col, 0.22), line=dict(width=0),
                                       name=f"{lbl}: predicted tube α<sub>j|k</sub>Ω" if key == "p"
                                            else f"{lbl}: predicted tube Ω",
                                       hoverinfo="skip"), row)
        _add(f"{key}_pred", go.Scatter(x=[], y=[], mode="lines",
                                       line=dict(color=col, width=1.2, dash="dot"),
                                       name=f"{lbl}: predicted nominal z<sub>j|k</sub>",
                                       hoverinfo="skip"), row)
        _add(f"{key}_past", go.Scatter(x=[], y=[], mode="lines",
                                       line=dict(color=col, width=2),
                                       name=f"{lbl}: true trajectory e<sub>y</sub>"), row)
        _add(f"{key}_tube", go.Scatter(x=[], y=[], mode="lines",
                                       line=dict(color=col, width=5),
                                       name=f"{lbl}: current tube α<sub>k</sub>Ω" if key == "p"
                                            else f"{lbl}: current tube Ω",
                                       hoverinfo="skip"), row)
        _add(f"{key}_viol", go.Scatter(x=[], y=[], mode="markers",
                                       marker=dict(symbol="x", size=9, color=col),
                                       name=f"{lbl}: constraint violation",
                                       showlegend=False), row)
        _add(f"{key}_car", go.Scatter(x=[], y=[], mode="markers",
                                      marker=dict(symbol="triangle-right", size=16, color=col,
                                                  line=dict(color="black", width=0.8)),
                                      name=f"{lbl}: vehicle", showlegend=False), row)
    _add("p_alpha", go.Scatter(x=[], y=[], mode="lines", line=dict(color=_HEX["p"], width=2),
                               name="α<sub>k</sub> HT-MPC"), 3)
    _add("r_alpha", go.Scatter(x=[], y=[], mode="lines", line=dict(color=_HEX["r"], width=2,
                               dash="dash"), name="α<sub>k</sub> RT-MPC (= 1)"), 3)
    _add("cursor", go.Scatter(x=[], y=[], mode="lines", line=dict(color="grey", width=1),
                              showlegend=False, hoverinfo="skip"), 3)

    # frames
    def _frame_data(k):
        data, idx = [], []
        for key, tr in (("p", tr_p), ("r", tr_r)):
            kk = min(k, len(tr.x) - 1)            # freeze after infeasibility
            if kk < 0:
                continue
            sx   = np.arange(kk + 1)*ds
            sh   = kk*ds + np.arange(N + 1)*ds
            zp   = tr.z_pred[kk, :, 0]
            ap   = tr.a_pred[kk] if key == "p" else np.ones(N + 1)
            band_x = np.concatenate([sh, sh[::-1]])
            band_y = np.concatenate([zp + ap*s, (zp - ap*s)[::-1]])
            a_now  = min(tr.alpha[kk], 1.) if key == "p" else 1.
            heading = -np.degrees(tr.x[kk, 1])
            vi = np.where(tr.viol_x[:kk + 1])[0]
            data += [
                go.Scatter(x=band_x, y=band_y),
                go.Scatter(x=sh, y=zp),
                go.Scatter(x=sx, y=tr.x[:kk + 1, 0]),
                go.Scatter(x=[kk*ds, kk*ds], y=[tr.z[kk, 0] - a_now*s, tr.z[kk, 0] + a_now*s]),
                go.Scatter(x=sx[vi], y=tr.x[vi, 0]),
                go.Scatter(x=[kk*ds], y=[tr.x[kk, 0]], marker=dict(angle=heading)),
            ]
            idx += [anim_idx[f"{key}_{nm}"] for nm in ("band", "pred", "past", "tube", "viol", "car")]
        kp = min(k, len(tr_p.x) - 1);  kr = min(k, len(tr_r.x) - 1)
        data += [go.Scatter(x=np.arange(kp + 1)*ds, y=tr_p.alpha[:kp + 1]),
                 go.Scatter(x=np.arange(kr + 1)*ds, y=tr_r.alpha[:kr + 1]),
                 go.Scatter(x=[k*ds, k*ds], y=[-0.05, 1.15])]
        idx += [anim_idx["p_alpha"], anim_idx["r_alpha"], anim_idx["cursor"]]
        return data, idx

    def _frame_title(k):
        t = k*Ts
        parts = [f"k = {k:3d}   t = {t:4.1f} s   x = {k*ds:6.1f} m"]
        kp = min(k, len(tr_p.x) - 1)
        parts.append(f"HT-MPC α<sub>k</sub> = {min(tr_p.alpha[kp], 1.):.3f}")
        if not tr_p.ok and k >= len(tr_p.x):
            parts.append(f"<b style='color:{_HEX['p']}'>HT-MPC infeasible at k = {len(tr_p.x)}</b>")
        if not tr_r.ok and k >= len(tr_r.x):
            parts.append(f"<b style='color:{_HEX['r']}'>RT-MPC infeasible at k = {len(tr_r.x)}</b>")
        return "   |   ".join(parts)

    frames = []
    for k in range(T_max):
        data, idx = _frame_data(k)
        frames.append(go.Frame(data=data, traces=idx, name=str(k),
                               layout=go.Layout(title_text=_frame_title(k))))
    fig.frames = frames
    d0, i0 = _frame_data(0)
    for tr_, i_ in zip(d0, i0):
        fig.data[i_].x = tr_.x;  fig.data[i_].y = tr_.y
        if tr_.marker is not None and tr_.marker.angle is not None:
            fig.data[i_].marker.angle = tr_.marker.angle

    lab_every = 5 if T_max <= 80 else 10
    steps = [dict(method="animate", label=(f"{k*Ts:.1f}" if k % lab_every == 0 else ""),
                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                                       transition=dict(duration=0))])
             for k in range(T_max)]
    fig.update_layout(
        height=height, template="plotly_white",
        title=dict(text=_frame_title(0), x=0.01, xanchor="left", yref="paper", y=1.0,
                   yanchor="bottom", pad=dict(b=28), font=dict(size=12)),
        legend=dict(orientation="h", y=-0.27, yanchor="top", x=0, font=dict(size=10)),
        margin=dict(l=60, r=20, t=90, b=160),
        updatemenus=[dict(type="buttons", showactive=False, x=1.0, xanchor="right",
                          y=1.0, yanchor="bottom", pad=dict(b=22), direction="left",
                          buttons=[
                              dict(label="▶ Play", method="animate",
                                   args=[None, dict(frame=dict(duration=frame_ms, redraw=False),
                                                    fromcurrent=True, transition=dict(duration=0))]),
                              dict(label="❚❚ Pause", method="animate",
                                   args=[[None], dict(mode="immediate",
                                                      frame=dict(duration=0, redraw=False),
                                                      transition=dict(duration=0))]),
                          ])],
        sliders=[dict(active=0, steps=steps, x=0.0, len=1.0, y=-0.09, yanchor="top",
                      pad=dict(t=0, b=0), ticklen=3, minorticklen=0,
                      font=dict(size=9),
                      currentvalue=dict(prefix="t = ", suffix=" s", visible=True,
                                        xanchor="left", font=dict(size=11)))],
    )
    ylim = [-W - 0.4, W + 0.4]
    fig.update_yaxes(title_text="e<sub>y</sub> [m]", range=ylim, row=1, col=1)
    fig.update_yaxes(title_text="e<sub>y</sub> [m]", range=ylim, row=2, col=1)
    fig.update_yaxes(title_text="α<sub>k</sub>", range=[-0.05, 1.15], row=3, col=1)
    fig.update_xaxes(range=[0, x_end], row=3, col=1, title_text="longitudinal distance [m]")
    for row in (1, 2):
        fig.add_annotation(x=0.5, y=-W + 0.15, text="slow lane", showarrow=False, xanchor="left",
                           font=dict(size=10, color=_HEX["constr"]), row=row, col=1)
        fig.add_annotation(x=0.5, y=W - 0.15, text="fast lane", showarrow=False, xanchor="left",
                           font=dict(size=10, color=_HEX["constr"]), row=row, col=1)
    if title:
        fig.layout.annotations[0].text = f"HT-MPC (homothetic tube, proposed) — {title}"
    return fig


# ── Plotly road-view animation (chase camera, vehicles drawn to scale) ───────

_CAR_L, _CAR_W = 4.5, 1.8          # vehicle footprint [m]


def _car_polygon(x, y, psi, L=_CAR_L, W=_CAR_W, front_only=False):
    """Closed polygon of a rectangle centred at (x, y), rotated by psi [rad]."""
    if front_only:
        pts = np.array([[L/2 - 1.0, -W/2], [L/2, -W/2], [L/2, W/2], [L/2 - 1.0, W/2]])
    else:
        pts = np.array([[-L/2, -W/2], [L/2, -W/2], [L/2, W/2], [-L/2, W/2]])
    c, s = np.cos(psi), np.sin(psi)
    R = np.array([[c, -s], [s, c]])
    p = pts@R.T + np.array([x, y])
    p = np.vstack([p, p[:1]])
    return p[:, 0], p[:, 1]


def animate_road(d: Design, tr_p: Trace, tr_r: Trace, y_ref_sig, title=None,
                 frame_ms: int = 100, height: int = 720, window: float = 60.0):
    """
    Road view: two road strips (HT-MPC on top, RT-MPC below), a chase camera that follows
    each vehicle, the vehicle drawn to scale and rotated by its heading error, the current
    tube cross-section alpha_k*Omega as a shaded patch around the nominal position, the
    predicted nominal path z_{j|k} with its tube ahead, and alpha_k in a third panel.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    Ts, vx0, N, s, W = d.Ts, d.vx0, d.N, d.s_ey, d.lane_width
    T_full = len(y_ref_sig)
    T_max  = max(len(tr_p.x), len(tr_r.x))
    ds     = Ts*vx0
    x_end  = (T_full - 1 + N)*ds + window
    s_ref  = np.arange(T_full + N)*ds
    y_ref_ext = np.concatenate([y_ref_sig, np.full(N, y_ref_sig[-1])])
    back, ahead = 0.2*window, 0.8*window

    fig = make_subplots(rows=3, cols=1, row_heights=[0.36, 0.36, 0.28],
                        vertical_spacing=0.10, shared_xaxes=False,
                        subplot_titles=("HT-MPC (homothetic tube, proposed)",
                                        "RT-MPC (fixed tube)",
                                        "tube scale α<sub>k</sub> along the track"))

    # ── static road furniture ────────────────────────────────────────────────
    for row in (1, 2):
        fig.add_shape(type="rect", x0=-back, x1=x_end, y0=-W, y1=W, row=row, col=1,
                      fillcolor="#e9e9e9", line=dict(width=0), layer="below")
        for yv, wdt, dash, col in ((W, 2.0, "solid", "#444"), (-W, 2.0, "solid", "#444"),
                                   (0., 1.2, "dash", "white"),
                                   (d.x_bounds[0], 0.8, "dot", _HEX["constr"]),
                                   (-d.x_bounds[0], 0.8, "dot", _HEX["constr"])):
            fig.add_shape(type="line", x0=-back, x1=x_end, y0=yv, y1=yv, row=row, col=1,
                          line=dict(color=col, width=wdt, dash=dash), layer="below")
        fig.add_trace(go.Scatter(x=s_ref, y=y_ref_ext, mode="lines", line_shape="hv",
                                 line=dict(color=_HEX["ref"], width=1), name="reference",
                                 showlegend=(row == 1), hoverinfo="skip"), row=row, col=1)
        fig.add_annotation(x=0.01, y=-W + 0.15, xref=f"x{'' if row == 1 else row} domain",
                           yref=f"y{'' if row == 1 else row}", text="slow lane",
                           showarrow=False, xanchor="left", yanchor="bottom",
                           font=dict(size=10, color="#555"))
        fig.add_annotation(x=0.01, y=W - 0.15, xref=f"x{'' if row == 1 else row} domain",
                           yref=f"y{'' if row == 1 else row}", text="fast lane",
                           showarrow=False, xanchor="left", yanchor="top",
                           font=dict(size=10, color="#555"))
    fig.add_shape(type="line", x0=0, x1=(T_full - 1)*ds, y0=1, y1=1, row=3, col=1,
                  line=dict(color="grey", width=0.6, dash="dot"))

    # ── animated traces (order matters, frames refer to these indices) ───────
    anim_idx = {}

    def _add(key, trace, row):
        anim_idx[key] = len(fig.data)
        fig.add_trace(trace, row=row, col=1)

    for key, row in (("p", 1), ("r", 2)):
        col, lbl = _HEX[key], LABELS[key]
        _add(f"{key}_band", go.Scatter(x=[], y=[], mode="lines", fill="toself",
                                       fillcolor=_rgba(col, 0.18), line=dict(width=0),
                                       name=f"{lbl}: predicted tube " +
                                            ("α<sub>j|k</sub>Ω" if key == "p" else "Ω"),
                                       hoverinfo="skip"), row)
        _add(f"{key}_pred", go.Scatter(x=[], y=[], mode="lines",
                                       line=dict(color=col, width=1.2, dash="dot"),
                                       name=f"{lbl}: predicted nominal z<sub>j|k</sub>",
                                       hoverinfo="skip"), row)
        _add(f"{key}_past", go.Scatter(x=[], y=[], mode="lines",
                                       line=dict(color=col, width=1.5),
                                       name=f"{lbl}: driven path", hoverinfo="skip"), row)
        _add(f"{key}_tube", go.Scatter(x=[], y=[], mode="lines", fill="toself",
                                       fillcolor=_rgba(col, 0.35), line=dict(width=0),
                                       name=f"{lbl}: current tube " +
                                            ("α<sub>k</sub>Ω" if key == "p" else "Ω"),
                                       hoverinfo="skip"), row)
        _add(f"{key}_car", go.Scatter(x=[], y=[], mode="lines", fill="toself",
                                      fillcolor=col, line=dict(color="black", width=1),
                                      name=f"{lbl}: vehicle", hoverinfo="skip"), row)
        _add(f"{key}_nose", go.Scatter(x=[], y=[], mode="lines", fill="toself",
                                       fillcolor="rgba(255,255,255,0.75)", line=dict(width=0),
                                       showlegend=False, hoverinfo="skip"), row)
        _add(f"{key}_viol", go.Scatter(x=[], y=[], mode="markers",
                                       marker=dict(symbol="x", size=10, color="red",
                                                   line=dict(width=2)),
                                       name=f"{lbl}: constraint violation",
                                       showlegend=False, hoverinfo="skip"), row)
    _add("p_alpha", go.Scatter(x=[], y=[], mode="lines", line=dict(color=_HEX["p"], width=2),
                               name="α<sub>k</sub> HT-MPC"), 3)
    _add("r_alpha", go.Scatter(x=[], y=[], mode="lines",
                               line=dict(color=_HEX["r"], width=2, dash="dash"),
                               name="α<sub>k</sub> RT-MPC (= 1)"), 3)
    _add("cursor", go.Scatter(x=[], y=[], mode="lines", line=dict(color="grey", width=1),
                              showlegend=False, hoverinfo="skip"), 3)

    # ── frames ───────────────────────────────────────────────────────────────
    def _frame(k):
        data, idx, ranges = [], [], {}
        for key, tr, ax in (("p", tr_p, "xaxis"), ("r", tr_r, "xaxis2")):
            kk = min(k, len(tr.x) - 1)
            xk = kk*ds
            sx = np.arange(kk + 1)*ds
            sh = xk + np.arange(N + 1)*ds
            zp = tr.z_pred[kk, :, 0]
            ap = tr.a_pred[kk] if key == "p" else np.ones(N + 1)
            a_now = min(tr.alpha[kk], 1.) if key == "p" else 1.
            zk = tr.z[kk, 0]
            ey, psi = tr.x[kk, 0], tr.x[kk, 1]
            vi = np.where(tr.viol_x[:kk + 1])[0]
            cx, cy = _car_polygon(xk, ey, psi)
            nx, ny = _car_polygon(xk, ey, psi, front_only=True)
            tube_x = [xk - _CAR_L/2, xk + _CAR_L/2, xk + _CAR_L/2, xk - _CAR_L/2, xk - _CAR_L/2]
            tube_y = [zk - a_now*s, zk - a_now*s, zk + a_now*s, zk + a_now*s, zk - a_now*s]
            data += [
                go.Scatter(x=np.concatenate([sh, sh[::-1]]),
                           y=np.concatenate([zp + ap*s, (zp - ap*s)[::-1]])),
                go.Scatter(x=sh, y=zp),
                go.Scatter(x=sx, y=tr.x[:kk + 1, 0]),
                go.Scatter(x=tube_x, y=tube_y),
                go.Scatter(x=cx, y=cy),
                go.Scatter(x=nx, y=ny),
                go.Scatter(x=sx[vi], y=tr.x[vi, 0]),
            ]
            idx += [anim_idx[f"{key}_{nm}"]
                    for nm in ("band", "pred", "past", "tube", "car", "nose", "viol")]
            ranges[ax] = dict(range=[xk - back, xk + ahead])
        kp = min(k, len(tr_p.x) - 1);  kr = min(k, len(tr_r.x) - 1)
        data += [go.Scatter(x=np.arange(kp + 1)*ds, y=tr_p.alpha[:kp + 1]),
                 go.Scatter(x=np.arange(kr + 1)*ds, y=tr_r.alpha[:kr + 1]),
                 go.Scatter(x=[k*ds, k*ds], y=[-0.05, 1.15])]
        idx += [anim_idx["p_alpha"], anim_idx["r_alpha"], anim_idx["cursor"]]
        return data, idx, ranges

    def _title(k):
        parts = [f"k = {k:3d}   t = {k*Ts:4.1f} s   x = {k*ds:6.1f} m"]
        kp = min(k, len(tr_p.x) - 1)
        parts.append(f"HT-MPC α<sub>k</sub> = {min(tr_p.alpha[kp], 1.):.3f}")
        if not tr_p.ok and k >= len(tr_p.x):
            parts.append(f"<b style='color:{_HEX['p']}'>HT-MPC infeasible at k = {len(tr_p.x)}</b>")
        if not tr_r.ok and k >= len(tr_r.x):
            parts.append(f"<b style='color:{_HEX['r']}'>RT-MPC infeasible at k = {len(tr_r.x)}</b>")
        return "   |   ".join(parts)

    frames = []
    for k in range(T_max):
        data, idx, ranges = _frame(k)
        frames.append(go.Frame(data=data, traces=idx, name=str(k),
                               layout=go.Layout(title_text=_title(k), **ranges)))
    fig.frames = frames
    d0, i0, r0 = _frame(0)
    for tr_, i_ in zip(d0, i0):
        fig.data[i_].x = tr_.x;  fig.data[i_].y = tr_.y

    lab_every = 5 if T_max <= 80 else 10
    steps = [dict(method="animate", label=(f"{k*Ts:.1f}" if k % lab_every == 0 else ""),
                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))])
             for k in range(T_max)]
    fig.update_layout(
        height=height, template="plotly_white",
        title=dict(text=_title(0), x=0.01, xanchor="left", yref="paper", y=1.0,
                   yanchor="bottom", pad=dict(b=28), font=dict(size=12)),
        legend=dict(orientation="h", y=-0.24, yanchor="top", x=0, font=dict(size=10)),
        margin=dict(l=60, r=20, t=90, b=120),
        updatemenus=[dict(type="buttons", showactive=False, x=1.0, xanchor="right",
                          y=1.0, yanchor="bottom", pad=dict(b=22), direction="left",
                          buttons=[
                              dict(label="▶ Play", method="animate",
                                   args=[None, dict(frame=dict(duration=frame_ms, redraw=True),
                                                    fromcurrent=True, transition=dict(duration=0))]),
                              dict(label="❚❚ Pause", method="animate",
                                   args=[[None], dict(mode="immediate",
                                                      frame=dict(duration=0, redraw=True),
                                                      transition=dict(duration=0))]),
                          ])],
        sliders=[dict(active=0, steps=steps, x=0.0, len=1.0, y=-0.06, yanchor="top",
                      pad=dict(t=0, b=0), ticklen=3, minorticklen=0, font=dict(size=9),
                      currentvalue=dict(prefix="t = ", suffix=" s", visible=True,
                                        xanchor="left", font=dict(size=11)))],
    )
    ylim = [-W - 0.6, W + 0.6]
    fig.update_yaxes(title_text="e<sub>y</sub> [m]", range=ylim, row=1, col=1, fixedrange=True)
    fig.update_yaxes(title_text="e<sub>y</sub> [m]", range=ylim, row=2, col=1, fixedrange=True)
    fig.update_yaxes(title_text="α<sub>k</sub>", range=[-0.05, 1.15], row=3, col=1)
    fig.update_xaxes(range=r0["xaxis"]["range"], row=1, col=1, showgrid=False,
                     zeroline=False, ticksuffix=" m")
    fig.update_xaxes(range=r0["xaxis2"]["range"], row=2, col=1, showgrid=False,
                     zeroline=False, ticksuffix=" m")
    fig.update_xaxes(range=[0, (T_full - 1)*ds], title_text="longitudinal position [m]",
                     row=3, col=1)
    if title:
        fig.layout.annotations[0].text = f"HT-MPC (homothetic tube, proposed) — {title}"
    return fig


animate = animate_road      # default view


# ── Static matplotlib summary figure ─────────────────────────────────────────

def plot_static(d: Design, tr_p: Trace, tr_r: Trace, y_ref_sig, title=None, show=True):
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    Ts, vx0, s, W = d.Ts, d.vx0, d.s_ey, d.lane_width
    T_p, T_r, T = len(tr_p.x), len(tr_r.x), len(y_ref_sig)
    x_p, x_r, x_ref = np.arange(T_p)*Ts*vx0, np.arange(T_r)*Ts*vx0, np.arange(T)*Ts*vx0
    a_p = np.clip(tr_p.alpha, 0., 1.)

    fig, (ax, ax_a, ax_u) = plt.subplots(
        3, 1, figsize=(8.0, 6.8), dpi=110, sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1., 1.], "hspace": 0.08})
    for yv, lw, ls, al in ((W, 1.0, "-", 1.), (-W, 1.0, "-", 1.),
                           (d.x_bounds[0], 0.6, "--", 0.6), (-d.x_bounds[0], 0.6, "--", 0.6),
                           (0., 0.5, ":", 0.35)):
        ax.axhline(yv, color=C["constr"], lw=lw, ls=ls, alpha=al)
    ax.fill_between(x_r, tr_r.z[:, 0]-s,     tr_r.z[:, 0]+s,     color=C["r"], alpha=0.15, lw=0.)
    ax.fill_between(x_p, tr_p.z[:, 0]-a_p*s, tr_p.z[:, 0]+a_p*s, color=C["p"], alpha=0.30, lw=0.)
    ax.step(x_ref, y_ref_sig, color=C["ref"], lw=0.8, where="post")
    ax.plot(x_r, tr_r.x[:, 0], color=C["r"], lw=1.0, ls="--")
    ax.plot(x_p, tr_p.x[:, 0], color=C["p"], lw=1.0, ls="--")
    for key, tr, xl in (("r", tr_r, x_r), ("p", tr_p, x_p)):
        vi = np.where(tr.viol_x)[0]
        if len(vi):
            ax.scatter(xl[vi], tr.x[vi, 0], c=C[key], s=28, marker="x", zorder=7)
        if not tr.ok and len(xl):
            for a_ in (ax, ax_a, ax_u):
                a_.axvline(xl[-1], color=C[key], lw=0.8, ls=":")
            ax_a.text(xl[-1], 0.55, f" {LABELS[key]} infeasible (k={len(xl)})",
                      color=C[key], fontsize=8, va="center", ha="left")
    ax.set_ylabel(r"$e_y$ [m]");  ax.set_ylim(-W-0.4, W+0.4)
    ax.text(x_ref[-1]*0.01, -W+0.12, "slow lane", color=C["constr"], fontsize=8, va="bottom")
    ax.text(x_ref[-1]*0.01,  W-0.12, "fast lane", color=C["constr"], fontsize=8, va="top")
    ax.legend(handles=[
        Line2D([0], [0], color=C["ref"], lw=0.8,          label="Reference"),
        Line2D([0], [0], color=C["p"],   lw=1.0, ls="--", label=LABELS["p"]),
        Line2D([0], [0], color=C["r"],   lw=1.0, ls="--", label=LABELS["r"]),
        Patch(facecolor=C["p"], alpha=0.30, label=r"$\alpha_k\Omega$ (HT-MPC)"),
        Patch(facecolor=C["r"], alpha=0.15, label=r"$\Omega$ (RT-MPC)"),
        Line2D([0], [0], color=C["constr"], lw=1.0,          label="Lane boundary"),
        Line2D([0], [0], color=C["constr"], lw=0.6, ls="--", label=r"$|e_y|$ constraint"),
    ], loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=4, frameon=False)
    ax_a.plot(x_r, tr_r.alpha, color=C["r"], lw=1.0, ls="--")
    ax_a.plot(x_p, tr_p.alpha, color=C["p"], lw=1.0)
    ax_a.axhline(1., color="0.5", lw=0.5, ls=":")
    ax_a.set_ylabel(r"$\alpha_k$");  ax_a.set_ylim(-0.05, 1.15);  ax_a.grid(True)
    ax_u.step(x_r, tr_r.u[:, 0], color=C["r"], lw=1.0, ls="--", where="post")
    ax_u.step(x_p, tr_p.u[:, 0], color=C["p"], lw=1.0,          where="post")
    for sgn in (+1, -1):
        ax_u.axhline(sgn*d.u_bounds[0], color=C["constr"], lw=0.6, ls="--", alpha=0.6)
    ax_u.set_ylabel(r"$\delta_s$ [rad]");  ax_u.set_xlabel("Longitudinal distance [m]")
    ax_u.set_xlim(0., x_ref[-1]);  ax_u.grid(True)
    if title:
        fig.suptitle(title, y=1.0, fontsize=9)
    if show:
        plt.show()
    return fig


# ── One-call convenience wrapper ──────────────────────────────────────────────

def run(d: Design, mode=MANOEUVRES[0], t_lc=1.2, t_hold=2.0, y_lane=1.75, T_sim=5.0,
        ramp=0.0, custom="0:-1.75; 1.2:1.75; 3.5:0.0", plant_seed=42, noise_seed=7):
    """Build the reference, simulate both controllers, return (tr_p, tr_r, y_ref, Delta_vx).
    Raises ValueError for an invalid manoeuvre."""
    wps = waypoints_for(mode, t_lc, t_hold, y_lane, T_sim, custom)
    y_ref_sig = build_reference(wps, T_sim, d.Ts, ramp, y_max=d.y_ref_max)
    tr_p, tr_r, Delta_vx = simulate(d, y_ref_sig, plant_seed, noise_seed)
    return tr_p, tr_r, y_ref_sig, Delta_vx
