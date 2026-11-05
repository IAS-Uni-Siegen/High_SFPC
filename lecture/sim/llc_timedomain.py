"""Exact (time-domain) steady state of the LLC and SRC converter, compared with the
first-harmonic approximation (FHA). Generates the data of the Lecture 9 figures.

Model (normalised): full-bridge input voltage +-1 (V_in = 1), L_r = 1, C_r = 1 (omega_r = 1, Z0 = 1),
L_m = Ln, ideal full-bridge rectifier into a constant output voltage M (n = 1), so that M = n V_o / V_in.
Rectifier modes: +1 (rectifier current i_r - i_m > 0, L_m sees +M), -1 (L_m sees -M),
0 (rectifier off: L_r and L_m in series resonate with C_r).
The load enters through Q = Z0 / R_e with R_e = 8 R_L / pi^2: for given F and Q the gain M is found
such that the average rectified current equals M / R_L = 8 Q M / pi^2.

Usage:  python llc_timedomain.py [modes|src|exact|all]
Runs much faster with numba installed (pip install numba); without numba it works, but slowly.
Output: ../fig/lec09/llc_modes.dat, src_plane.dat, llc_exact_q02.dat, llc_exact_q05.dat, llc_exact_q10.dat
"""
import os
import sys
import numpy as np
from scipy.optimize import brentq

try:
    from numba import njit
except ImportError:  # pure-Python fallback
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fig', 'lec09')


@njit(cache=True)
def deriv(x, vs, Vo, Ln, mode):
    """State derivative; x = [i_r, v_c, i_m]."""
    ir, vc = x[0], x[1]
    if mode == 0:  # rectifier off
        d = (vs - vc) / (1.0 + Ln)
        return np.array([d, ir, d])
    vm = Vo * mode
    return np.array([vs - vc - vm, ir, vm / Ln])


@njit(cache=True)
def step(x, vs, Vo, Ln, mode, dt):
    """One RK4 step with the mode logic of the ideal rectifier."""
    if mode == 0:
        vm = Ln * (vs - x[1]) / (1.0 + Ln)
        if vm >= Vo:
            mode = 1
        elif vm <= -Vo:
            mode = -1
    k1 = deriv(x, vs, Vo, Ln, mode)
    k2 = deriv(x + 0.5 * dt * k1, vs, Vo, Ln, mode)
    k3 = deriv(x + 0.5 * dt * k2, vs, Vo, Ln, mode)
    k4 = deriv(x + dt * k3, vs, Vo, Ln, mode)
    xn = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    isn = xn[0] - xn[2]
    if mode == 1 and isn <= 0.0:
        vm = Ln * (vs - xn[1]) / (1.0 + Ln)
        xn[2] = xn[0]
        mode = -1 if vm <= -Vo else 0
    elif mode == -1 and isn >= 0.0:
        vm = Ln * (vs - xn[1]) / (1.0 + Ln)
        xn[2] = xn[0]
        mode = 1 if vm >= Vo else 0
    return xn, mode


@njit(cache=True)
def run(F, Ln, Vo, nper, nstep, x0, mode0):
    """Integrates nper periods; returns the final state, mode and the rectified current
    averaged over the last period."""
    dt = 2 * np.pi / F / nstep
    x = x0.copy()
    mode = mode0
    Iavg = 0.0
    for p in range(nper):
        acc = 0.0
        for k in range(nstep):
            vs = 1.0 if k < nstep // 2 else -1.0
            xo = x
            x, mode = step(x, vs, Vo, Ln, mode, dt)
            acc += 0.5 * (abs(xo[0] - xo[2]) + abs(x[0] - x[2]))
        Iavg = acc / nstep
    return x, mode, Iavg


@njit(cache=True)
def record(F, Ln, Vo, nstep, x0, mode0):
    """One period of the steady state: theta, i_r, v_c, i_m."""
    dt = 2 * np.pi / F / nstep
    x = x0.copy()
    mode = mode0
    rec = np.zeros((nstep + 1, 4))
    rec[0, 1:] = x
    for k in range(nstep):
        vs = 1.0 if k < nstep // 2 else -1.0
        x, mode = step(x, vs, Vo, Ln, mode, dt)
        rec[k + 1, 0] = (k + 1) * dt * F
        rec[k + 1, 1:] = x
    return rec


def steady(F, Ln, M, nper=600, nstep=4000):
    x, mode, I = run(F, Ln, M, nper, nstep, np.zeros(3), 0)
    return x, mode, I


def M_exact(F, Ln, Q, nper=600):
    """Exact gain for the normalised frequency F and quality factor Q."""
    g = lambda M: steady(F, Ln, M, nper)[2] - 8 * Q * M / np.pi**2
    return brentq(g, 1e-3, 3.0, xtol=1e-4)


def M_fha(F, Ln, Q):
    return np.abs(Ln * F**2 / (((Ln + 1) * F**2 - 1) + 1j * (F**2 - 1) * F * Q * Ln))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    os.makedirs(OUT, exist_ok=True)
    Ln = 5.0
    if what in ('all', 'modes'):
        cols = []
        for F in (0.8, 1.0, 1.3):
            M = M_exact(F, Ln, 0.5)
            x, mode, _ = steady(F, Ln, M)
            cols.append(record(F, Ln, M, 4000, x, mode))
            print(f"modes: F = {F}, M = {M:.4f}")
        with open(os.path.join(OUT, 'llc_modes.dat'), 'w') as fh:
            fh.write('# Normalised LLC (Ln = 5, Q = 0.5): one period from the rising edge of the bridge voltage.\n')
            fh.write('# theta in rad; currents in units of V_in/Z0 for f_sw/f_r = 0.8, 1.0 and 1.3 (exact time-domain steady state)\n')
            fh.write('th ir08 im08 ir10 im10 ir13 im13\n')
            for k in range(0, 4001, 20):
                fh.write(f"{cols[0][k, 0]:.4f} " + " ".join(f"{c[k, 1]:.4f} {c[k, 3]:.4f}" for c in cols) + "\n")
    if what in ('all', 'src'):
        LnS = 1e7  # SRC: magnetising inductance practically infinite
        M = M_exact(1.2, LnS, 1.0)
        x, mode, _ = steady(1.2, LnS, M)
        r = record(1.2, LnS, M, 4000, x, mode)
        print(f"SRC F = 1.2, Q = 1: exact M = {M:.4f}, FHA {M_fha(1.2, LnS, 1.0):.4f}")
        with open(os.path.join(OUT, 'src_plane.dat'), 'w') as fh:
            fh.write(f'# SRC state plane, F = 1.2, Q = 1, exact M = {M:.4f}: m_c = v_c/V_in, j = Z0 i/V_in over one period\n')
            fh.write('mc j\n')
            for k in range(0, 4001, 10):
                fh.write(f"{r[k, 2]:.4f} {r[k, 1]:.4f}\n")
    if what in ('all', 'exact'):
        for Q, Fmin in ((0.2, 0.42), (0.5, 0.5), (1.0, 0.62)):
            Fs = np.round(np.concatenate([np.arange(Fmin, 1.0, 0.04), np.arange(1.0, 2.01, 0.1)]), 3)
            with open(os.path.join(OUT, f'llc_exact_q{int(Q * 10):02d}.dat'), 'w') as fh:
                fh.write(f'# LLC Ln = 5, Q = {Q}: exact (time-domain) gain versus FHA\nF Mex Mfha\n')
                for F in Fs:
                    Me = M_exact(F, Ln, Q)
                    fh.write(f"{F:.3f} {Me:.4f} {M_fha(F, Ln, Q):.4f}\n")
                    print(f"Q = {Q}, F = {F:.3f}: exact {Me:.4f}, FHA {M_fha(F, Ln, Q):.4f}", flush=True)
