"""One-seed diagnostic that applies the same measures to every rule, to compare the rules that work (hebb with its
own basal threshold, calcium, bcm) with the apically gated ones that do not (us, burst, gated_hebb). The rule is
wrapped so that its raw (unbounded) update dw is seen at every time step; because x_in is binary, dw / lr_bas on any
active row is the rule's per-neuron post-synaptic factor at that step. Per block of trials it reports:

  - where potentiation and depression go: tone / texture / distractor rows, and at which steps (tone step, texture
    step, distractor-only steps)
  - across-neuron concentration of texture-row potentiation: participation ratio (effective number of neurons) and
    the number of neurons that receive 80 percent of it
  - whether the sign of the post-synaptic factor follows the neuron's own drive: Spearman correlation across neurons
    between the factor and x_bas at texture, tone and distractor-only steps, and the fraction of neurons potentiated
  - within-neuron target: share of a neuron's potentiation landing on its currently strongest synapse
  - the resulting weights: detectors, total weight, texture selectivity

Usage (from the repo root; the config is a JSON dict of hyperparameters, the seed defaults to 0):
    python -m sandbox.diag_compare '{"rule": "us", "bound": "l1", "lr_bas": 0.0005, ...}' 0
"""
import sys, json, numpy as np, torch
from scipy.stats import spearmanr
from l5apical.helper import *
import sandbox.rules as R
from sandbox.simulate import simulate_seed_bu, K_W_BAS_H, K_W_BAS_T, K_W_AP, expert_trial_proxy, detector_counts, texture_selectivity

hp = json.loads(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
BLOCK = int(hp.pop('_block', 300))
base = R.RULES[hp['rule']]
STEPS = OUTCOME_T - TONE_T + 1  # 17 stimulus-period calls + 1 outcome call per trial
tex_t = TEXTURE_T - TONE_T
CLS = ['tone', 'texture', 'distractor']
STEP = ['tone_step', 'texture_step', 'distractor_step']
LOG = dict(calls=0)
blocks = {}


def blk(trial):
    b = trial // BLOCK
    if b not in blocks:
        blocks[b] = dict(pot=np.zeros((3, 3)), dep=np.zeros((3, 3)), pot_neuron_tex=None, pot_neuron_all=None,
                         rho=dict((k, []) for k in STEP), frac_pos=dict((k, []) for k in STEP),
                         pot_argmax=0., pot_total=0., n=0)
    return blocks[b]


def wrapped(w_bas, x_in, x_som, x_bas, x_ap, gains, thetas, hp_, state):
    dw = base(w_bas, x_in, x_som, x_bas, x_ap, gains, thetas, hp_, state)
    c = LOG['calls']; LOG['calls'] += 1
    t, trial = c % STEPS, c // STEPS
    if dw is None or t == STEPS - 1 or x_in.sum() == 0:
        return dw
    B = blk(trial)
    if B['pot_neuron_tex'] is None:
        B['pot_neuron_tex'] = np.zeros(w_bas.shape[1]); B['pot_neuron_all'] = np.zeros(w_bas.shape[1])
    d = dw.numpy()
    active = x_in.numpy() > 0
    step = 0 if active[0] else (1 if active[1] or active[2] else 2)
    rows = np.where(active)[0]
    cls = np.where(rows == 0, 0, np.where(rows <= 2, 1, 2))
    pos, neg = np.clip(d[rows], 0, None), np.clip(-d[rows], 0, None)
    for k in range(3):
        B['pot'][k, step] += pos[cls == k].sum(); B['dep'][k, step] += neg[cls == k].sum()
    B['pot_neuron_all'] += pos.sum(0)
    if step == 1:
        B['pot_neuron_tex'] += pos[cls == 1].sum(0)
    post = d[rows[0]] / hp_['lr_bas']
    xb = x_bas.numpy()
    if np.ptp(post) > 0:
        B['rho'][STEP[step]].append(spearmanr(post, xb).correlation)
    B['frac_pos'][STEP[step]].append(float((post > 0).mean()))
    # share of potentiation landing on the neuron's currently strongest synapse
    amax = w_bas.numpy().argmax(0)
    on_max = (rows[:, None] == amax[None, :])
    B['pot_argmax'] += pos[on_max].sum(); B['pot_total'] += pos.sum()
    B['n'] += 1
    return dw


R.RULES[hp['rule']] = wrapped
r = simulate_seed_bu(hp, seed=seed)
correct = r[K_CORRECT_TRIALS]
print(json.dumps(hp), 'seed', seed)
print(f"expert proxy {expert_trial_proxy(correct)}  diverged {r.get('diverged_at')}  final perf {np.nanmean(correct[-200:]):.3f}")
w_h, w_t = r[K_W_BAS_H], r[K_W_BAS_T]
w0 = w_h[0]
tex0 = w0[1:3].max(0)          # initial strongest texture weight per neuron
strongest0 = w0.max(0)         # initial strongest weight per neuron
for b in sorted(blocks):
    B = blocks[b]
    lo, hi = b * BLOCK, min((b + 1) * BLOCK, len(correct))
    pot, dep = B['pot'], B['dep']
    tot = pot.sum() + dep.sum()
    i = int(np.searchsorted(w_t, hi - 1)); i = min(i, len(w_t) - 1)
    W = w_h[i]
    det = detector_counts(W)
    print(f"\n--- trials {lo}-{hi}  perf {np.nanmean(correct[lo:hi]):.2f}  |  raw update mass: pot {pot.sum() / tot:.2f} dep {dep.sum() / tot:.2f}")
    print("  potentiation by row (tone/tex/dist) x step (tone/tex/dist-only), share of all potentiation:")
    for k in range(3):
        print(f"    {CLS[k]:10s} " + ' '.join(f"{pot[k, s] / max(pot.sum(), 1e-12):5.2f}" for s in range(3)) + f"   | depression share " + ' '.join(f"{dep[k, s] / max(dep.sum(), 1e-12):5.2f}" for s in range(3)))
    pn, pa = B['pot_neuron_tex'], B['pot_neuron_all']
    pr = lambda v: (v.sum() ** 2 / max((v ** 2).sum(), 1e-24)) if v.sum() > 0 else 0.
    n80 = lambda v: int(np.searchsorted(np.cumsum(np.sort(v)[::-1]), 0.8 * v.sum()) + 1) if v.sum() > 0 else 0
    rho_tex = spearmanr(pn, tex0).correlation if pn.sum() > 0 and np.ptp(pn) > 0 else np.nan
    rho_str = spearmanr(pa, strongest0).correlation if pa.sum() > 0 and np.ptp(pa) > 0 else np.nan
    rho_ap = spearmanr(pn, r[K_W_AP][hi - 1]).correlation if pn.sum() > 0 and np.ptp(pn) > 0 else np.nan
    print(f"  texture-row potentiation across neurons: effective n {pr(pn):6.1f}, neurons holding 80% {n80(pn):3d}, "
          f"rho with initial texture weight {rho_tex:+.2f}, with apical weight {rho_ap:+.2f}")
    print(f"  all potentiation across neurons:         effective n {pr(pa):6.1f}, neurons holding 80% {n80(pa):3d}, "
          f"rho with initial strongest weight {rho_str:+.2f}")
    print("  post factor vs neuron's own drive x_bas (Spearman across neurons) and fraction of neurons potentiated:")
    for s in STEP:
        rh, fp = B['rho'][s], B['frac_pos'][s]
        print(f"    {s:16s} rho {np.nanmean(rh) if rh else np.nan:+.2f}   frac>0 {np.mean(fp) if fp else np.nan:.2f}")
    print(f"  share of potentiation on the neuron's currently strongest synapse {B['pot_argmax'] / max(B['pot_total'], 1e-12):.2f}")
    print(f"  weights at trial {w_t[i]}: detectors tone {det['det_tone']} texture {det['det_texture']} distractor {det['det_distractor']}, "
          f"w_sum {W.sum(0).mean():.2f}, texture selectivity {texture_selectivity(W).mean():.2f}, "
          f"median strongest w {np.median(W.max(0)):.2f}, tone w mean {W[0].mean():.3f}")
