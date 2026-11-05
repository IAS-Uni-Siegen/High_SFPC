import os
import numpy as np
# ripple cancellation of N interleaved buck phases: K = dI_sum / dI_phase versus D
def ripple(N, D, n=4000):
    t = np.arange(n)/n
    def tri(t):                     # one phase, normalised V_in = L = T = 1, V_o = D
        t = t % 1.0
        return np.where(t < D, (1-D)*t, (1-D)*D - D*(t-D))
    s = sum(tri(t + k/N) for k in range(N))
    return (s.max() - s.min())/(D*(1-D))
Ds = np.round(np.arange(0.02, 0.981, 0.01), 3)
rows = []
for D in Ds:
    K = [ripple(N, D) for N in (2, 3, 4)]
    m = [np.floor(N*D) for N in (2, 3, 4)]
    Kf = [N*(D - mm/N)*((mm+1)/N - D)/(D*(1-D)) for N, mm in zip((2, 3, 4), m)]
    rows.append((D, *K, *Kf))
rows = np.array(rows)
print("max deviation numeric vs formula:", np.abs(rows[:,1:4]-rows[:,4:7]).max())
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fig', 'lec11', 'interleave.dat'), 'w') as fh:
    fh.write('# Ripple of the summed inductor current of N interleaved buck phases relative to one phase, versus D\n')
    fh.write('# K = N (D - m/N)((m+1)/N - D) / (D (1 - D)), m = floor(N D); checked numerically (interleave.py)\n')
    fh.write('D K2 K3 K4\n')
    for r in rows:
        fh.write(f"{r[0]:.2f} {r[4]:.4f} {r[5]:.4f} {r[6]:.4f}\n")
print(rows[[8, 23, 48],:4])
