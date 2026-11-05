"""Charge-multiplier analysis of two-phase switched-capacitor converters (Seeman and Sanders 2008).
Netlist: caps = [(name, n_plus, n_minus)], sw = [(name, n_a, n_b, phase)], nodes 'in', 'out', 'gnd'.
The input is an ideal source V_in, the output an ideal source V_out (large output capacitor)."""
import numpy as np

def analyse(caps, sw, Vin=1.0):
    nodes = sorted({n for c in caps for n in c[1:3]} | {n for s in sw for n in s[1:3]} | {'in', 'out', 'gnd'})
    idx = {n: i for i, n in enumerate(nodes)}
    nc, ns, nn = len(caps), len(sw), len(nodes)
    # ---------- charges: unknowns q_c^(1), q_c^(2) (into + terminal), q_s (in its phase, from a to b), q_in^(j), q_out^(j)
    nu = 2*nc + ns + 4
    rows, rhs = [], []
    for ph in (1, 2):
        for n in nodes:
            if n == 'gnd':
                continue
            r = np.zeros(nu)
            for i, (_, p, m, *_) in enumerate(caps):
                col = (ph-1)*nc + i
                if p == n: r[col] -= 1      # charge leaving node into the + plate
                if m == n: r[col] += 1      # charge arriving from the - plate
            for k, (_, a, b, phs) in enumerate(sw):
                if phs != ph: continue
                col = 2*nc + k
                if a == n: r[col] -= 1
                if b == n: r[col] += 1
            if n == 'in':  r[2*nc + ns + (ph-1)] += 1          # source delivers q_in into node
            if n == 'out': r[2*nc + ns + 2 + (ph-1)] -= 1      # q_out leaves the node into the load
            rows.append(r); rhs.append(0.0)
    for i in range(nc):                                         # charge balance of every capacitor
        r = np.zeros(nu); r[i] = 1; r[nc+i] = 1; rows.append(r); rhs.append(0.0)
    r = np.zeros(nu); r[2*nc+ns+2] = 1; r[2*nc+ns+3] = 1; rows.append(r); rhs.append(1.0)   # q_out per period = 1
    A, b = np.array(rows), np.array(rhs)
    q, res, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < nu:
        raise ValueError(f"charge flow not unique (rank {rank} of {nu})")
    ac = np.abs(q[:nc]); ar = np.abs(q[2*nc:2*nc+ns]); qin = q[2*nc+ns] + q[2*nc+ns+1]
    # ---------- voltages: unknowns phi^(1), phi^(2) for all nodes, v_c, V_out
    nv = 2*nn + nc + 1
    rows, rhs = [], []
    for ph in (1, 2):
        o = (ph-1)*nn
        r = np.zeros(nv); r[o+idx['gnd']] = 1; rows.append(r); rhs.append(0.0)
        r = np.zeros(nv); r[o+idx['in']] = 1; rows.append(r); rhs.append(Vin)
        r = np.zeros(nv); r[o+idx['out']] = 1; r[2*nn+nc] = -1; rows.append(r); rhs.append(0.0)
        for i, (_, p, m, *_) in enumerate(caps):
            r = np.zeros(nv); r[o+idx[p]] = 1; r[o+idx[m]] = -1; r[2*nn+i] = -1; rows.append(r); rhs.append(0.0)
        for k, (_, a, bb, phs) in enumerate(sw):
            if phs != ph: continue
            r = np.zeros(nv); r[o+idx[a]] = 1; r[o+idx[bb]] = -1; rows.append(r); rhs.append(0.0)
    A, b = np.array(rows), np.array(rhs)
    v, res, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if np.linalg.norm(A @ v - b) > 1e-9:
        raise ValueError("inconsistent voltages")
    Vout = v[2*nn+nc]; vc = np.abs(v[2*nn:2*nn+nc])
    vr = []
    for k, (_, a, bb, phs) in enumerate(sw):
        o = (2-phs)*nn                      # the other phase: switch open
        vr.append(abs(v[o+idx[a]] - v[o+idx[bb]]))
    vr = np.array(vr)
    return dict(ratio=Vin/Vout, Vout=Vout, ac=ac, ar=ar, vc=vc, vr=vr, qin=qin,
                cap_va=np.sum(ac*vc)/Vout, sw_va=np.sum(ar*vr)/Vout,
                kSSL=np.sum(ac**2), kFSL=2*np.sum(ar**2))

def series_parallel(N):
    caps = [(f'C{i}', f't{i}', f'b{i}') for i in range(1, N)]
    sw = [('S_in', 'in', 't1', 1)]
    for i in range(1, N-1):
        sw.append((f'S_s{i}', f'b{i}', f't{i+1}', 1))
    sw.append(('S_o', f'b{N-1}', 'out', 1))
    for i in range(1, N):
        sw += [(f'S_t{i}', f't{i}', 'out', 2), (f'S_b{i}', f'b{i}', 'gnd', 2)]
    return caps, sw

def ladder(N):
    L = ['gnd', 'out'] + [f'L{k}' for k in range(2, N)] + ['in']
    caps = [(f'Cd{k}', L[k+1], L[k]) for k in range(1, N-1)]
    caps += [(f'Cf{k}', f'R{k}', f'R{k-1}') for k in range(1, N)]
    sw = [(f'S1_{k}', f'R{k}', L[k], 1) for k in range(N)] + [(f'S2_{k}', f'R{k}', L[k+1], 2) for k in range(N)]
    return caps, sw

def dickson4():
    caps = [('C1', 'n1', 'A'), ('C2', 'n2', 'B'), ('C3', 'n3', 'A')]
    sw = [('SA1', 'A', 'gnd', 1), ('SB1', 'B', 'out', 1), ('SA2', 'A', 'out', 2), ('SB2', 'B', 'gnd', 2),
          ('S1', 'out', 'n1', 1), ('S3', 'n2', 'n3', 1), ('S2', 'n1', 'n2', 2), ('S4', 'n3', 'in', 2)]
    return caps, sw

def doubler4():
    caps = [('Cf1', 't1', 'b1'), ('Cm', 'm', 'gnd'), ('Cf2', 't2', 'b2')]
    sw = [('S1a', 'in', 't1', 1), ('S1b', 'b1', 'm', 1), ('S1c', 't1', 'm', 2), ('S1d', 'b1', 'gnd', 2),
          ('S2a', 'm', 't2', 1), ('S2b', 'b2', 'out', 1), ('S2c', 't2', 'out', 2), ('S2d', 'b2', 'gnd', 2)]
    return caps, sw

if __name__ == '__main__':
    for name, (caps, sw) in [('SP 2:1', series_parallel(2)), ('SP 3:1', series_parallel(3)), ('SP 4:1', series_parallel(4)),
                             ('ladder 3:1', ladder(3)), ('ladder 4:1', ladder(4)), ('Dickson 4:1', dickson4()), ('doubler 4:1', doubler4())]:
        r = analyse(caps, sw)
        print(f"{name:12s} ratio {r['ratio']:.3f}  caps {len(caps)} sw {len(sw)}  a_c {np.round(r['ac'],3)}  v_c {np.round(r['vc']/r['Vout'],2)}")
        print(f"{'':12s} a_r {np.round(r['ar'],3)}  v_r {np.round(r['vr']/r['Vout'],2)}")
        print(f"{'':12s} sum a_c^2 = {r['kSSL']:.4f}, 2 sum a_r^2 = {r['kFSL']:.4f}, cap VA = {r['cap_va']:.3f} P, switch VA = {r['sw_va']:.3f} P, q_in = {r['qin']:.3f}")
