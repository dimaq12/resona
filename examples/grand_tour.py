"""
grand_tour.py — the whole theory, one chained pipeline.
==============================================================================
resona is not a bag of tricks — it is ONE object (the operator's spectral
response) seen through its two conjugate representations, with everything else a
consequence.  This tour chains the modules so each feeds the next, and shows they
are mutually consistent.

  RESPONSE  μ_B  (moments, COMPOSES)        ⟂      RESOLVENT  G(z)  (poles, RESOLVES)
     resona.of / moment / free / beta      Lanczos       wkernel / subordination / flow
                         └──────────  LIFT linearizes composition  ──────────┘
                         (lift: R/S-transform, Carleman) ;  DEFECT = SHOCK = EDGE = COST

Run:  python3 examples/grand_tour.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy import linalg
import resona

rng = np.random.default_rng(0)


def banner(s): print("\n" + "═" * 74 + f"\n{s}\n" + "═" * 74)


if __name__ == "__main__":
    banner("ACT 1 — COMPOSE A⊞B without ever forming A+B  (response closure)")
    M = 900
    A = np.diag(rng.uniform(-1.5, 1.5, M))                              # structured spectrum
    Q, _ = linalg.qr(rng.standard_normal((M, M))); B = Q @ A @ Q.T       # a FREE copy
    sA = resona.of(lambda v: A @ v, M, k=40, probes=16)                 # PROBE
    sB = resona.of(lambda v: B @ v, M, k=40, probes=16)
    print(f"  PROBE:    support(A) = [{sA.extreme()[0]:+.2f}, {sA.extreme()[1]:+.2f}]"
          f"   (structured, uniform spectrum)")
    fd = resona.free.freeness_defect(lambda v: A @ v, lambda v: B @ v, M, probes=20)
    print(f"  FREE:     freeness defect = {fd:.4f}  ⇒ A,B free ⇒ composition CLOSES")
    m_pred = resona.lift.free_convolution(sA, sB, order=2)               # COMPOSE (no joint matvec)
    m_true = [np.trace(np.linalg.matrix_power(A + B, n)) / M for n in range(1, 3)]
    print(f"  COMPOSE:  moments m₁,m₂ of A⊞B from μ_A,μ_B ALONE — max|Δm| = "
          f"{max(abs(p-t) for p,t in zip(m_pred, m_true)):.4f}  (no joint matvec)")
    s = np.sqrt(m_pred[1] - m_pred[0] ** 2)                              # free σ
    lev = resona.beta.beta_spectrum(m_pred[0] - 2 * s, m_pred[0] + 2 * s,
                                    m_pred[0], m_pred[1], M)             # READ as a spectrum
    mae = np.mean(np.abs(np.sort(lev) - np.sort(linalg.eigvalsh(A + B)))) / (4 * s)
    print(f"  BETA:     reconstructed spectrum of A⊞B vs eig(A+B): MAE = {mae*100:.1f}%")

    banner("ACT 2 — the SAME composition is a FLOW with a SHOCK  (lift ⟂ resolvent)")
    wg = np.linspace(0.05, 0.4, 6)
    RA = resona.lift.r_transform(sA, wg); RB = resona.lift.r_transform(sB, wg)
    sAB = resona.of(lambda v: (A + B) @ v, M, k=40, probes=8)
    RAB = resona.lift.r_transform(sAB, wg)
    print(f"  LIFT:     R_(A⊞B) = R_A + R_B  →  rel.err = "
          f"{np.max(np.abs(RAB-(RA+RB)))/np.max(np.abs(RAB)):.4f}  (shock LINEAR in R)")
    N = 600
    A2 = np.diag(np.concatenate([-np.ones(N // 2), np.ones(N // 2)]))    # two bands ±1
    s2 = resona.of(lambda v: A2 @ v, N, k=80, probes=6)
    tc = resona.flow.shock_time(s2)
    print(f"  FLOW:     bands ±1 under μ_t=μ_0⊞sc(t) MERGE at t_c ≈ {tc:.2f}  (exact 1.0)")
    print(f"  SUBORD.:  ⟨DOS⟩ of A+σ·GOE in closed form (Pastur) — m₂ = "
          f"{np.trapezoid(np.linspace(-3,3,1500)**2 * resona.subordination.averaged_dos(s2,0.5,np.linspace(-3,3,1500),2e-3), np.linspace(-3,3,1500)):.3f}"
          f"  (= m₂(A)+σ² = 1.25)")
    print("            └─ the band merger IS the defect = spectral edge = Burgers shock.")

    banner("ACT 3 — the conjugate (eigenvector) side, and the cost of the answer")
    n = 140; H = rng.standard_normal((n, n)); A0 = (H + H.T) / 2
    Bs = [np.diag((np.arange(n) == i).astype(float)) for i in range(0, n, 20)]
    _, V = linalg.eigh(A0)
    W = resona.wkernel.wkernel(V[:, :len(Bs)], Bs)                       # ∂λ/∂k (conjugate of Φ)
    dk = resona.wkernel.design(W, np.full(len(Bs), 0.05))               # inverse design
    print(f"  WKERNEL:  W=∂λ/∂k built; design solves W·dk=target to "
          f"{np.max(np.abs(W@dk-np.full(len(Bs),0.05))):.1e}")
    lim = resona.defect.richardson_limit([np.pi + 1.3/k + 0.7/k**2 for k in (25, 50, 100, 200)],
                                         (25, 50, 100, 200))
    print(f"  DEFECT:   Richardson tableau on a 1/n sequence → π to {abs(lim-np.pi):.1e}")
    x = np.arange(300)
    ep, _ = resona.cost.is_extractable(np.sin(2 * np.pi * x / 7))
    eg, _ = resona.cost.is_extractable(np.array([pow(3, int(i), 100003*100019) for i in x], float))
    print(f"  COST:     extraction dial — periodic: {'EXTRACTABLE' if ep else 'wall'};"
          f"  aˣ mod N: {'extractable' if eg else 'GENUINE WALL'}")

    banner("ONE OBJECT, MANY FACES")
    print("  • The response μ_B (moments) COMPOSES — A⊞B from the measures alone,")
    print("    legitimate exactly when the parts are FREE (freeness defect → 0).")
    print("  • The resolvent G (poles) RESOLVES — W-kernel, subordination, the flow;")
    print("    Lanczos is the bridge between the two pictures.")
    print("  • The LIFT (R/S-transform, Carleman) makes composition LINEAR.")
    print("  • DEFECT = SHOCK = spectral EDGE = the COST singularity — one locus,")
    print("    where free probability's own fixed point goes critical.")
    print("  Eight modules, one conjugate-pair theory.")
    print("═" * 74)
