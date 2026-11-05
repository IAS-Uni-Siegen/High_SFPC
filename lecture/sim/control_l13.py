"""Data of the Lecture 13 figures: buck control-to-output response, compensated loop gain with and
without digital delay, and the inductor current of peak current-mode control with and without slope
compensation. Usage: python control_l13.py  ->  ../fig/lec13/*.dat"""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fig', 'lec13')
os.makedirs(OUT, exist_ok=True)

# buck 48 V -> 12 V, 100 W, 1 MHz, L = 2.2 uH, C = 20 uF (ESR 2 mOhm), R = 1.44 Ohm, ramp V_M = 1 V
Vin, Vo, fsw, L, C, Resr, R, VM = 48.0, 12.0, 1e6, 2.2e-6, 20e-6, 2e-3, 1.44, 1.0
f = np.logspace(2, np.log10(5e5), 600)
s = 2j * np.pi * f
Zc = Resr + 1 / (s * C)
Zload = 1 / (1 / R + 1 / Zc)
Gvd = Vin * Zload / (s * L + Zload)                 # control-to-output, per unit duty
f0 = 1 / (2 * np.pi * np.sqrt(L * C)); Q = R * np.sqrt(C / L)
# type III compensator: two zeros at 0.8 f0, two poles at fsw/2, crossover 100 kHz
fz, fp, fc = 0.8 * f0, fsw / 2, 1e5
def Gc_shape(s):
    return (1 + s / (2 * np.pi * fz))**2 / (s * (1 + s / (2 * np.pi * fp))**2)
sc = 2j * np.pi * fc
ZLc = 1 / (1 / R + 1 / (Resr + 1 / (sc * C)))
Gvd_c = Vin * ZLc / (sc * L + ZLc)
wI = 1 / abs(Gc_shape(sc) * Gvd_c / VM)
T = wI * Gc_shape(s) * Gvd / VM
Td = 1.5 / fsw
Tdel = T * np.exp(-s * Td)
def margin(Tf):
    mag = np.abs(Tf); k = np.where(mag < 1)[0][0]
    fcx = np.interp(0, [-20*np.log10(mag[k-1]), -20*np.log10(mag[k])], [f[k-1], f[k]])
    ph = np.interp(fcx, f, np.unwrap(np.angle(Tf)) * 180 / np.pi)
    return fcx, 180 + ph
fcx, pm = margin(T); fcd, pmd = margin(Tdel)
print(f"f0 = {f0/1e3:.2f} kHz, Q = {Q:.2f}, ESR zero {1/(2*np.pi*Resr*C)/1e6:.2f} MHz")
print(f"loop: fc = {fcx/1e3:.1f} kHz, PM = {pm:.1f} deg; with 1.5 Ts delay: fc = {fcd/1e3:.1f} kHz, PM = {pmd:.1f} deg; delay lag at fc {360*fcx*Td:.1f} deg")
with open(os.path.join(OUT, 'buck_bode.dat'), 'w') as fh:
    fh.write('# Buck 48 to 12 V, L = 2.2 uH, C = 20 uF (ESR 2 mOhm), R = 1.44 Ohm: control-to-output Gvd (V per unit duty)\n')
    fh.write('# and loop gain T with type III compensator (zeros at 0.8 f0, poles at fsw/2, fc = 100 kHz), Td = 1.5 Ts\n')
    fh.write('f Gdb Gph Tdb Tph TDdb TDph\n')
    unwrap = lambda x: np.unwrap(np.angle(x)) * 180 / np.pi
    gph, tph, tdph = unwrap(Gvd), unwrap(T), unwrap(Tdel)
    for k in range(0, len(f), 3):
        fh.write(f"{f[k]:.1f} {20*np.log10(abs(Gvd[k])):.3f} {gph[k]:.2f} {20*np.log10(abs(T[k])):.3f} {tph[k]:.2f} "
                 f"{20*np.log10(abs(Tdel[k])):.3f} {tdph[k]:.2f}\n")

# peak current-mode control, buck 12 V -> 7.2 V (D = 0.6), L = 2.2 uH, 1 MHz
Vin2, Vo2, L2, T2 = 12.0, 7.2, 2.2e-6, 1e-6
m1, m2 = (Vin2 - Vo2) / L2, Vo2 / L2           # A/s
Ic = 5.0
def simulate(ma, n=16, di0=0.15):
    # steady state valley current for the reference Ic: Ic - ma*D*T = i_v + m1*D*T
    D = Vo2 / Vin2
    iv = Ic - ma * D * T2 - m1 * D * T2
    i = iv + di0
    pts = [(0.0, i)]
    for k in range(n):
        t0 = k * T2
        ton = (Ic - i) / (m1 + ma)               # i + m1 t = Ic - ma t
        ton = min(max(ton, 0.0), 0.95 * T2)
        ip = i + m1 * ton
        pts.append((t0 + ton, ip))
        i = ip - m2 * (T2 - ton)
        pts.append((t0 + T2, i))
    return np.array(pts)
a = simulate(0.0); b = simulate(0.5 * m2)
print(f"ratio without slope compensation {-m2/m1:.2f}, with ma = m2/2: {-(m2-0.5*m2)/(m1+0.5*m2):.2f}")
with open(os.path.join(OUT, 'subharm.dat'), 'w') as fh:
    fh.write('# Peak current mode, buck 12 V -> 7.2 V (D = 0.6), L = 2.2 uH, 1 MHz, Ic = 5 A, initial valley offset 0.15 A\n')
    fh.write('# t in us; i0: no slope compensation, i5: slope compensation ma = m2/2\n')
    fh.write('t i0 i5\n')
    for (ta, ia), (tb, ib) in zip(a, b):
        fh.write(f"{ta*1e6:.4f} {ia:.4f} {ib:.4f}\n")
print('note: breakpoints of the two cases differ in time; written separately below')
with open(os.path.join(OUT, 'subharm0.dat'), 'w') as fh:
    fh.write('t i\n'); [fh.write(f"{t*1e6:.4f} {i:.4f}\n") for t, i in a]
with open(os.path.join(OUT, 'subharm5.dat'), 'w') as fh:
    fh.write('t i\n'); [fh.write(f"{t*1e6:.4f} {i:.4f}\n") for t, i in b]
os.remove(os.path.join(OUT, 'subharm.dat'))
# boost RHP zero example: 12 V -> 48 V, 100 W, L = 4.7 uH
D = 0.75; Rb = 48**2 / 100; Lb = 4.7e-6
print(f"boost RHP zero: {(1-D)**2*Rb/(2*np.pi*Lb)/1e3:.1f} kHz")
