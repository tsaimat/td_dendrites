"""Per-configuration summary of a sandbox.scan_gated (or scan_smith) JSON-lines file.
Usage: python -m sandbox.tab_gated <file.jsonl> [more files] [--prot fresh|inh] [--sort expert|name]
Columns: seeds run, seeds that reached expert (Smith), seeds diverged, mean +- sd of the Smith expert trial over the
seeds that reached it (after the lift in the inhibition protocol), mean performance of the last 200 trials, and the
means of the weight metrics at the end (and at the lift): texture / tone detectors (percent of neurons), total input
per neuron, mean |w - w(trial 0)|, and where recorded the task inputs' share of all basal weight (percent; 9 at the
start) and the inputs above 0.05 per neuron (7.4 at the start)."""
import json, sys, collections
import numpy as np


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p)]
    return rows


def summarize(rows, prot):
    by = collections.OrderedDict()
    for r in rows:
        if r['prot'] == prot:
            by.setdefault(r['name'], []).append(r)
    out = []
    for name, rs in by.items():
        ex = np.array([r['expert'] for r in rs if r['expert'] is not None], float)
        m = lambda k: np.nanmean([r.get(k, np.nan) for r in rs]) if any(k in r for r in rs) else np.nan
        out.append(dict(name=name, n=len(rs), n_exp=len(ex), n_div=sum(r['diverged'] is not None for r in rs),
                        exp_mean=ex.mean() if len(ex) else np.nan, exp_sd=ex.std() if len(ex) else np.nan,
                        perf=m('final_perf'), det_tx=m('det_texture_end'), det_tone=m('det_tone_end'),
                        w_tot=m('w_total_end'), dw=m('dw_abs_end'), det_tx_lift=m('det_texture_lift'),
                        dw_lift=m('dw_abs_lift'), task=m('task_share_end'), k_in=m('n_inputs_end')))
    return out


if __name__ == '__main__':
    args = sys.argv[1:]
    prot = args[args.index('--prot') + 1] if '--prot' in args else 'fresh'
    sort = args[args.index('--sort') + 1] if '--sort' in args else 'expert'
    paths = [a for i, a in enumerate(args)
             if not a.startswith('--') and (i == 0 or not args[i - 1].startswith('--'))]
    out = summarize(load(paths), prot)
    if sort == 'expert':
        out.sort(key=lambda d: (d['n'] - d['n_exp'], d['exp_mean'] if np.isfinite(d['exp_mean']) else 1e9))
    lift = prot == 'inh'
    print(f"{'name':40s} {'n':>3s} {'exp':>3s} {'div':>3s} {'expert':>13s} {'perf':>5s} {'det_tx':>6s} {'tone':>5s} "
          f"{'w_tot':>5s} {'dw':>6s} {'task%':>5s} {'k_in':>4s}"
          + (f" {'tx@lift':>7s} {'dw@lift':>7s}" if lift else ''))
    for d in out:
        e = f"{d['exp_mean']:6.0f} +- {d['exp_sd']:4.0f}" if np.isfinite(d['exp_mean']) else f"{'-':>13s}"
        print(f"{d['name']:40s} {d['n']:3d} {d['n_exp']:3d} {d['n_div']:3d} {e:>13s} {d['perf']:5.2f} "
              f"{d['det_tx']:6.1f} {d['det_tone']:5.1f} {d['w_tot']:5.2f} {d['dw']:6.3f} {d['task']:5.1f} "
              f"{d['k_in']:4.1f}"
              + (f" {d['det_tx_lift']:7.1f} {d['dw_lift']:7.3f}" if lift else ''))
