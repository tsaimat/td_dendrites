"""One-seed diagnostic of the apically gated rules (us, burst): logs the post-synaptic factor (gain - g_bar or P - Pbar)
of every neuron at the texture time step split by texture, where the update mass goes (tone / texture / distractor
rows), and how the weights of the 20 initially most texture-selective neurons evolve. Usage (from the repo root):
    python -m sandbox.diag_gated '{"rule": "us", "bound": "l1", "lr_bas": 0.0005, "lr_ap": 0.064, "us_tau": 5, "lr_policy": 0.128, "lr_td": 0.032, "n_z": 203, "init_power": 12}' 0
"""
import sys, json, numpy as np, torch
from l5apical.helper import *
import sandbox.rules as R
from sandbox.simulate import simulate_seed_bu, K_W_BAS_H, expert_trial_proxy

hp = json.loads(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
base = R.RULES[hp['rule']]
texture_t = TEXTURE_T - TONE_T
LOG = dict(post_tex=[], gbar_tex=[], dw_rows=[], t=0, trial=0)

def wrapped(w_bas, x_in, x_som, x_bas, x_ap, gains, thetas, hp_, state):
    dw = base(w_bas, x_in, x_som, x_bas, x_ap, gains, thetas, hp_, state)
    t = LOG['t']
    if t == texture_t:
        key = 'g_bar' if 'g_bar' in state else 'p_bar'
        ref = state[key]
        post = (gains - ref) if key == 'g_bar' else (x_ap + hp_.get('burst_kappa', 0.) * x_som - ref)
        LOG['post_tex'].append(post.numpy().copy()); LOG['gbar_tex'].append(ref.numpy().copy())
        LOG['tex'].append(int(x_in[T1_IDX].item()))
    if dw is not None:
        a = dw.abs()
        LOG['dw_rows'].append([a[0].sum().item(), a[1:3].sum().item(), a[3:].sum().item()])
    LOG['t'] = (t + 1) % (OUTCOME_T - TONE_T + 1)  # 17 stimulus-period calls + 1 outcome call per trial
    return dw
LOG['tex'] = []
R.RULES[hp['rule']] = wrapped
r = simulate_seed_bu({**hp, 'store_full': True}, seed=seed)
print('expert proxy', expert_trial_proxy(r[K_CORRECT_TRIALS]), 'diverged', r.get('diverged_at'))
post = np.array(LOG['post_tex']); tex = np.array(LOG['tex']); gb = np.array(LOG['gbar_tex'])
print('n texture-time samples', len(post), ' T1 trials', tex.sum())
for lo, hi in [(0, 300), (600, 900), (1200, 1500)]:
    p = post[lo:hi]; tx = tex[lo:hi]
    print(f'trials {lo}-{hi}: post factor at texture time, mean over neurons: T1 {p[tx==1].mean():.4f}  T2 {p[tx==0].mean():.4f} '
          f'| max neuron {p.mean(0).max():.3f} | g_bar/p_bar mean {gb[lo:hi].mean():.3f} min {gb[lo:hi].min():.3f} max {gb[lo:hi].max():.3f}')
    # per neuron: does the post factor differ between T1 and T2 trials?
    d = p[tx==1].mean(0) - p[tx==0].mean(0)
    print(f'   per-neuron T1-T2 difference of post factor: mean {d.mean():+.5f}, max |d| {np.abs(d).max():.5f}, vs mean post {p.mean():.4f}')
dwr = np.array(LOG['dw_rows']); n_tr = len(r[K_CORRECT_TRIALS])
per_trial = dwr.reshape(-1, dwr.shape[0] // n_tr, 3).sum(1) if dwr.shape[0] % n_tr == 0 else None
tot = dwr.sum(0); print('share of |dw| mass: tone %.3f  textures %.3f  distractors %.3f' % tuple(tot / tot.sum()))
w = r[K_W_BAS_H]; ts = [v for k, v in r.items() if isinstance(v, np.ndarray) and v.ndim == 1 and v.dtype.kind == 'i' and len(v) == len(w)][0]
sel = lambda W: np.abs(W[T1_IDX] - W[T2_IDX]) / (W[T1_IDX] + W[T2_IDX] + 1e-12)
w0 = w[0]; top = np.argsort(-(w0[1:3].max(0)))[:20]
print('20 neurons with the strongest initial texture weight:')
for i in range(0, len(ts), max(1, len(ts) // 6)):
    W = w[i]
    print(f'  trial {ts[i]:5d}: max texture w {W[1:3].max(0)[top].mean():.3f}  other texture w {W[1:3].min(0)[top].mean():.3f}  '
          f'selectivity {sel(W)[top].mean():.3f}  tone w {W[0][top].mean():.3f}  distractor sum {W[3:].sum(0)[top].mean():.3f}  '
          f'| all neurons: selectivity {sel(W).mean():.3f}, texture sum {W[1:3].sum(0).mean():.3f}')
