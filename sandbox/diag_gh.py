"""One-seed diagnostic of gated_hebb: the post-synaptic factor gate * (post - theta) at tone and texture time (sign,
by texture, fraction of neurons potentiated), where the potentiation and depression mass goes (tone / texture /
distractor rows), and how the weights of the 20 initially most texture-selective neurons evolve. Usage:
    python -m sandbox.diag_gh '{"rule": "gated_hebb", "bound": "soft", "lr_bas": 0.0005, "lr_ap": 0.064, "lr_policy": 0.128, "lr_td": 0.032, "n_z": 203, "init_power": 12}' 0
"""
import sys, json, numpy as np, torch
from l5apical.helper import *
import sandbox.rules as R
from sandbox.simulate import simulate_seed_bu, K_W_BAS_H, K_W_BAS_T, expert_trial_proxy, detector_counts

hp = json.loads(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
base = R.RULES[hp['rule']]
tone_t, texture_t, n_calls = 0, TEXTURE_T - TONE_T, OUTCOME_T - TONE_T + 1
LOG = dict(t=0, tex=[], post=dict(tone=[], tex=[]), gate=dict(tone=[], tex=[]), pot=[], dep=[], trial_pot=[])

def wrapped(w_bas, x_in, x_som, x_bas, x_ap, gains, thetas, hp_, state):
    dw = base(w_bas, x_in, x_som, x_bas, x_ap, gains, thetas, hp_, state)
    t = LOG['t']
    gate = R.apical_norm(x_ap).pow(hp_.get('gate_power', 1.0))
    post = x_bas if hp_.get('gh_post', 'som') == 'bas' else x_som
    factor = (gate * (post - hp_.get('gh_theta_scale', 1.0) * thetas)).numpy()
    if t == tone_t:
        LOG['post']['tone'].append(factor.copy()); LOG['gate']['tone'].append(gate.numpy().copy())
    if t == texture_t:
        LOG['post']['tex'].append(factor.copy()); LOG['gate']['tex'].append(gate.numpy().copy())
        LOG['tex'].append(int(x_in[T1_IDX].item()))
    if dw is not None:
        p, d = dw.clamp(min=0), (-dw).clamp(min=0)
        LOG['pot'].append([p[0].sum().item(), p[1:3].sum().item(), p[3:].sum().item()])
        LOG['dep'].append([d[0].sum().item(), d[1:3].sum().item(), d[3:].sum().item()])
    LOG['t'] = (t + 1) % n_calls
    return dw
R.RULES[hp['rule']] = wrapped
r = simulate_seed_bu({**hp, 'store_full': True, 'snapshot_every': 100}, seed=seed)
n_tr = len(r[K_CORRECT_TRIALS])
print('expert proxy', expert_trial_proxy(r[K_CORRECT_TRIALS]), 'diverged', r.get('diverged_at'))
tex = np.array(LOG['tex'])
for lo, hi in [(0, 300), (300, 600), (600, 900), (900, 1200), (1200, 1500), (1500, 1800)]:
    if lo >= len(tex): break
    line = f'trials {lo:4d}-{hi:4d}:'
    for k in ('tone', 'tex'):
        f = np.array(LOG['post'][k])[lo:hi]; g = np.array(LOG['gate'][k])[lo:hi]
        line += (f' | {k}: gate {g.mean():.3f}, factor mean {f.mean():+.4f}, frac neurons >0 {np.mean(f.mean(0) > 0):.2f}'
                 f' (n {int(np.sum(f.mean(0) > 0))}), max {f.mean(0).max():+.4f}')
    f = np.array(LOG['post']['tex'])[lo:hi]; tx = tex[lo:hi]
    d = f[tx == 1].mean(0) - f[tx == 0].mean(0)
    line += f' | tex factor T1-T2 per neuron: mean {d.mean():+.5f} max|d| {np.abs(d).max():.5f}'
    print(line)
pot, dep = np.array(LOG['pot']), np.array(LOG['dep'])
calls = len(pot) // n_tr if len(pot) % n_tr == 0 else None
print('update mass (sum |dw| before bound) rows tone / textures / distractors:')
print('  potentiation %.3e %.3e %.3e   shares %.3f %.3f %.3f' % (*pot.sum(0), *(pot.sum(0) / max(pot.sum(), 1e-12))))
print('  depression   %.3e %.3e %.3e   shares %.3f %.3f %.3f' % (*dep.sum(0), *(dep.sum(0) / max(dep.sum(), 1e-12))))
if calls:
    pt, dt = pot.reshape(n_tr, calls, 3).sum(1), dep.reshape(n_tr, calls, 3).sum(1)
    for lo, hi in [(0, 300), (600, 900), (1200, 1500)]:
        if lo >= n_tr: break
        print(f'  trials {lo}-{hi}: pot per trial tone/tex/dis {pt[lo:hi].mean(0)[0]:.2e} {pt[lo:hi].mean(0)[1]:.2e} {pt[lo:hi].mean(0)[2]:.2e}'
              f'  dep {dt[lo:hi].mean(0)[0]:.2e} {dt[lo:hi].mean(0)[1]:.2e} {dt[lo:hi].mean(0)[2]:.2e}')
w, ts = r[K_W_BAS_H], r[K_W_BAS_T]
sel = lambda W: np.abs(W[T1_IDX] - W[T2_IDX]) / (W[T1_IDX] + W[T2_IDX] + 1e-12)
w0 = w[0]; top = np.argsort(-(w0[1:3].max(0)))[:20]
print('20 neurons with the strongest initial texture weight, and population:')
for i in range(0, len(ts), max(1, len(ts) // 6)):
    W = w[i]
    if np.isnan(W).all(): continue
    dc = detector_counts(W)
    print(f'  trial {ts[i]:5d}: max texture w {W[1:3].max(0)[top].mean():.3f}  other texture w {W[1:3].min(0)[top].mean():.3f}  '
          f'selectivity {sel(W)[top].mean():.3f}  tone w {W[0][top].mean():.3f}  distractor sum {W[3:].sum(0)[top].mean():.3f}  '
          f'| all: selectivity {sel(W).mean():.3f}, texture sum {W[1:3].sum(0).mean():.3f}, tone {W[0].mean():.3f}, '
          f'w_sum {W.sum(0).mean():.3f}, detectors {dc}')
