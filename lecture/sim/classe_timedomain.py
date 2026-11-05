"""Exact periodic steady state of the class E inverter (ideal switch, linear capacitance)
and the data of the Lecture 10 figures.

Normalisation: V_DC = 1, omega = 1 (period 2 pi), nominal load R = 1.
Circuit: V_DC - L_F (choke) - drain node; switch and C_sh from drain to ground;
series branch L - C_0 - R from drain to ground. Q_L = omega L / R with L the whole series
inductance (convention of Sokal, QEX 2001); C_0 sets the excess reactance X = omega L - 1/(omega C_0).
States x = [i_F, v_d, i_0, v_C0]. The switch is on for 0 <= theta < 2 pi D. When the drain
voltage tries to go below zero the body diode clamps it (same equations as the on state).

Usage:  python classe_timedomain.py
Output: ../fig/lec10/classe_ideal.dat, classe_load.dat, classe_ql.dat, phi2_z.dat
"""
import os
import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fig', 'lec10')
D = 0.5


def matrices(QL, b, xr, R=1.0, LFr=1e4):
    """Augmented state matrices (5x5) for the off state and the on/clamped state.
    b = omega C_sh R_nom, xr = X / R_nom (both for the nominal R_nom = 1), R = actual load."""
    L = QL                        # omega L / R_nom with omega = R_nom = 1
    C0 = 1.0 / (QL - xr)          # 1/(omega C0) = QL - xr
    C, LF = b, LFr
    Aoff = np.zeros((5, 5))
    Aoff[0, 1], Aoff[0, 4] = -1 / LF, 1 / LF
    Aoff[1, 0], Aoff[1, 2] = 1 / C, -1 / C
    Aoff[2, 1], Aoff[2, 2], Aoff[2, 3] = 1 / L, -R / L, -1 / L
    Aoff[3, 2] = 1 / C0
    Aon = Aoff.copy()
    Aon[1, :] = 0.0               # v_d held at zero
    Aon[0, 1] = 0.0
    Aon[2, 1] = 0.0
    return Aoff, Aon


def linear_ss(QL, b, xr, R=1.0, LFr=1e4):
    """Steady state without diode conduction; returns the state at the end of the off interval."""
    Aoff, Aon = matrices(QL, b, xr, R, LFr)
    P0 = np.eye(5)
    P0[1, 1] = 0.0                # switch closes: drain capacitor discharged
    M = expm(Aoff * 2 * np.pi * (1 - D)) @ expm(Aon * 2 * np.pi * D) @ P0
    x = np.linalg.solve(np.eye(4) - M[:4, :4], M[:4, 4])
    return x


def zvs_conditions(p, QL, LFr):
    b, xr = p
    x = linear_ss(QL, b, xr, 1.0, LFr)
    return [x[1], x[0] - x[2]]    # v_d = 0 and dv_d/dt = 0 at turn-on


def waveform(QL, b, xr, R=1.0, LFr=1e4, n=3600, x_end=None):
    """One period with diode clamp, from the state x_end at the end of the previous period
    (steady state found by shooting). Returns theta, v_d, i_S (switch current, drain to source)."""
    Aoff, Aon = matrices(QL, b, xr, R, LFr)
    dt = 2 * np.pi / n
    Eoff, Eon = expm(Aoff * dt), expm(Aon * dt)

    def period(x0, record=False):
        z = np.append(x0, 1.0)
        z[1] = 0.0
        th, v, i = [0.0], [0.0], [z[0] - z[2]]
        clamped = False
        for k in range(n):
            on = k < int(round(D * n))
            if on or clamped:
                z = Eon @ z
                if not on and z[0] - z[2] >= 0.0:   # diode current ends: capacitor takes over
                    clamped = False
            else:
                z = Eoff @ z
                if z[1] < 0.0:                       # body diode clamps the drain
                    z[1] = 0.0
                    clamped = True
            if record:
                th.append((k + 1) * dt)
                v.append(z[1])
                i.append(z[0] - z[2] if (on or clamped) else 0.0)
        return z[:4], np.array(th), np.array(v), np.array(i)

    if x_end is None:
        x_end = linear_ss(QL, b, xr, R, LFr)
    x_end = fsolve(lambda x: period(x)[0] - x, x_end, xtol=1e-11)
    _, th, v, i = period(x_end, True)
    return th, v, i, x_end


def design(QL, LFr=1e4):
    """Shunt capacitance and excess reactance for ZVS and ZDS at Q_L; returns a dict."""
    b, xr = fsolve(zvs_conditions, [0.1836, 1.1525], args=(QL, LFr), xtol=1e-12)
    th, v, i, x = waveform(QL, b, xr, 1.0, LFr)
    Aoff, Aon = matrices(QL, b, xr, 1.0, LFr)
    # output power from the input current: P = V_DC * mean(i_F); recompute i_F on the grid
    n = len(th) - 1
    dt = 2 * np.pi / n
    Eoff, Eon = expm(Aoff * dt), expm(Aon * dt)
    z = np.append(x, 1.0)
    z[1] = 0.0
    iF, i0 = [z[0]], [z[2]]
    for k in range(n):
        z = (Eon if k < int(round(D * n)) else Eoff) @ z
        iF.append(z[0]); i0.append(z[2])
    iF, i0 = np.array(iF), np.array(i0)
    P = np.trapezoid(iF, th) / (2 * np.pi)
    P_R = np.trapezoid(i0**2, th) / (2 * np.pi)   # R = 1
    Idc = P
    return dict(QL=QL, b=b, xr=xr, P=P, P_R=P_R, RP=P, Vpk=v.max(), Ipk=i.max() / Idc,
                th=th, v=v, i=i / Idc, x=x)


def sokal_R(QL):
    """Sokal (QEX 2001) fit for R P / V^2, for comparison."""
    return 0.576801 * (1.0000086 - 0.414395 / QL - 0.557501 / QL**2 + 0.205967 / QL**3)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    # 1) ideal class E (Q_L -> infinity): analytical waveforms
    a = np.sqrt(1 + np.pi**2 / 4)
    phi = np.arctan2(1, -np.pi / 2)
    th = np.linspace(0, 2 * np.pi, 721)
    v = np.where(th < np.pi, 0.0, np.pi * ((th - np.pi) + a * (np.cos(th + phi) - np.cos(np.pi + phi))))
    i = np.where(th < np.pi, 1 - a * np.sin(th + phi), 0.0)
    io = a * np.sin(th + phi)           # output current / I_DC into the load network (i_S = I_DC - i_o)
    with open(os.path.join(OUT, 'classe_ideal.dat'), 'w') as fh:
        fh.write('# Ideal class E, D = 0.5, Q_L -> infinity: switch voltage v/V_DC, switch current i/I_DC,\n')
        fh.write('# output current i_o/I_DC; theta in degrees from turn-on (analytical, Raab 1977)\n')
        fh.write('th v i io\n')
        for k in range(0, 721, 2):
            fh.write(f"{np.degrees(th[k]):.1f} {v[k]:.4f} {i[k]:.4f} {io[k]:.4f}\n")
    print(f"ideal: Vpk = {v.max():.4f} at {np.degrees(th[v.argmax()]):.1f} deg, "
          f"Ipk = {i.max():.4f} at {np.degrees(th[i.argmax()]):.1f} deg, i(pi-) = {1 + a * np.sin(phi):.4f}")
    # 2) finite Q_L design table
    rows = []
    for QL in (3.0, 5.0, 10.0, 20.0, 1e3):
        d = design(QL)
        rows.append(d)
        print(f"QL = {QL:7.1f}: wCR = {d['b']:.4f}, X/R = {d['xr']:.4f}, RP/V^2 = {d['RP']:.4f} "
              f"(P_R {d['P_R']:.4f}, Sokal {sokal_R(QL):.4f}), Vpk = {d['Vpk']:.3f}, Ipk/Idc = {d['Ipk']:.3f}")
    with open(os.path.join(OUT, 'classe_ql.dat'), 'w') as fh:
        fh.write('# Class E design for ZVS and ZDS versus Q_L = omega L/R (L: whole series inductance), D = 0.5,\n')
        fh.write('# choke omega L_F = 1e4 R (exact time-domain steady state, classe_timedomain.py)\n')
        fh.write('QL wCR XR RPV2 Vpk Ipk\n')
        for d in rows:
            fh.write(f"{d['QL']:.0f} {d['b']:.4f} {d['xr']:.4f} {d['RP']:.4f} {d['Vpk']:.3f} {d['Ipk']:.3f}\n")
    # 3) load sensitivity: design at Q_L = 10 for R = 1, operate at other R
    d = design(10.0)
    cols = []
    for Rf in (0.6, 1.0, 1.6):
        th, v, i, x = waveform(10.0, d['b'], d['xr'], Rf)
        cols.append((th, v, i))
        print(f"R/Rnom = {Rf}: v at turn-on {x[1]:.3f}, Vpk {v.max():.3f}")
    with open(os.path.join(OUT, 'classe_load.dat'), 'w') as fh:
        fh.write('# Class E designed for Q_L = 10 at R_nom, operated at R = 0.6, 1.0 and 1.6 R_nom:\n')
        fh.write('# drain voltage v/V_DC versus theta (degrees from turn-on), body diode clamps at 0\n')
        fh.write('th v06 v10 v16\n')
        for k in range(0, len(cols[0][0]), 10):
            fh.write(f"{np.degrees(cols[0][0][k]):.1f} " + " ".join(f"{c[1][k]:.4f}" for c in cols) + "\n")
    # 4) class Phi2 input network (Rivas et al. 2008): peaks at f_s and 3 f_s, null at 2 f_s
    #    normalised: omega_s = 1, C_F = 1; inductor Q = 100 for finite peaks
    LF, CMR, LMR, Qi = 1 / 2.25, 15 / 16, 4 / 15, 100.0
    x = np.logspace(np.log10(0.2), np.log10(5), 801)
    s = 1j * x
    ZLF = s * LF + x * LF / Qi
    ZMR = s * LMR + x * LMR / Qi + 1 / (s * CMR)
    Z = 1 / (1 / ZLF + s * 1.0 + 1 / ZMR)
    Zc = 1 / (s * 1.0)                    # C_F alone, for comparison
    with open(os.path.join(OUT, 'phi2_z.dat'), 'w') as fh:
        fh.write('# Class Phi2 input network, normalised omega_s = 1, C_F = 1: L_F = 1/2.25, C_MR = 15/16, L_MR = 4/15,\n')
        fh.write('# inductor Q = 100; |Z| in units of 1/(omega_s C_F); Zc: C_F alone\n')
        fh.write('F Z Zc\n')
        for k in range(len(x)):
            fh.write(f"{x[k]:.4f} {abs(Z[k]):.4f} {abs(Zc[k]):.4f}\n")
    print("Phi2 |Z| at 1, 2, 3:", [round(abs(Z[np.argmin(abs(x - f))]), 2) for f in (1, 2, 3)])
