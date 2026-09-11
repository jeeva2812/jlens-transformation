"""Exploratory reanalysis; historical test set is NOT a fresh holdout.

Select layer by exit-only CV, report all layers and paired base-model controls.
Never load the large dense activation arrays. No model inference required.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rh.transfer import fit


def auc(y, s):
    p, n = s[y == 1], s[y == 0]
    return float(((p[:, None] > n).sum() + .5 * (p[:, None] == n).sum()) / (len(p)*len(n)))


def cv(X, y):
    rng = np.random.default_rng(0)
    folds = [[] for _ in range(5)]
    for label in (0, 1):
        for i, group in enumerate(np.array_split(rng.permutation(np.where(y == label)[0]), 5)):
            folds[i].extend(group.tolist())
    scores = np.zeros(len(y))
    for test in folds:
        train = np.setdiff1d(np.arange(len(y)), test)
        mu, w = fit(X[train], y[train])
        scores[test] = (X[test] - mu) @ w
    return auc(y, scores)


def main():
    root = Path('out/rh')
    index = {r['task_id']: r for r in map(json.loads, (root/'index.jsonl').read_text().splitlines())}
    rows = []
    layers = list(range(0,32,4))
    for m in map(json.loads, (root/'acts/manifest.jsonl').read_text().splitlines()):
        r = index[m['task_id']]
        if r.get('artifact_label') not in ('exploit_only','exploit_plus_substantive_attempt') or m['kind'] not in ('exit','always_equal'):
            continue
        paths = [root / part / m['file'] for part in ('acts','acts_base')]
        if not all(p.exists() for p in paths):
            continue
        with np.load(paths[0]) as a, np.load(paths[1]) as b:
            if not np.array_equal(a['tokens'], b['tokens']):
                raise ValueError('Paired tokens differ')
            tags_a, tags_b = list(a['key_tag']), list(b['key_tag'])
            if 'exploit+4' not in tags_a or 'exploit+4' not in tags_b:
                continue
            ia, ib = tags_a.index('exploit+4'), tags_b.index('exploit+4')
            assert a['key_pos'][ia] == b['key_pos'][ib]
            rows.append(dict(task_id=m['task_id'], kind=m['kind'], y=int(r['artifact_label']=='exploit_plus_substantive_attempt'),
                             adapted=a['key'][ia,layers].astype(np.float32), base=b['key'][ib,layers].astype(np.float32)))
    y = np.array([r['y'] for r in rows]); train = np.array([r['kind']=='exit' for r in rows]); test = ~train
    result = dict(n_train=int(train.sum()), n_test=int(test.sum()), n_train_positive=int(y[train].sum()), n_test_positive=int(y[test].sum()),
                  label='Judge-labeled substantive attempt later in transcript; not correctness or honesty',
                  caveat='Exploratory reused historical test set; paired base is teacher-forced on adapted transcripts', methods={})
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(9,5))
    for method in ('adapted','base'):
        X = np.stack([r[method] for r in rows])
        records=[]
        for k,l in enumerate(layers):
            mu,w=fit(X[train,k],y[train]); scores=(X[test,k]-mu)@w
            records.append(dict(layer=l, train_cv_auc=cv(X[train,k],y[train]), transfer_auc=auc(y[test],scores), scores=scores.tolist()))
        chosen=max(records,key=lambda r:r['train_cv_auc'])
        scores=np.array(chosen['scores']); yt=y[test]
        boot=[]
        for _ in range(5000):
            ids=np.concatenate([rng.choice(np.where(yt==c)[0],sum(yt==c),replace=True) for c in (0,1)])
            boot.append(auc(yt[ids],scores[ids]))
        null=[auc(rng.permutation(yt),scores) for _ in range(5000)]
        result['methods'][method]=dict(layers=records, selected_layer=chosen['layer'], selected_auc=chosen['transfer_auc'],
            stratified_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(), permutation_p=(1+sum(v>=chosen['transfer_auc'] for v in null))/5001)
        ax.plot(layers,[r['transfer_auc'] for r in records],'-o',label=method+' (same transcript prefix)')
        ax.scatter([chosen['layer']],[chosen['transfer_auc']],s=180,facecolors='none',edgecolors='black',zorder=5)
        print(method, result['methods'][method]['selected_layer'], result['methods'][method]['selected_auc'],result['methods'][method]['stratified_bootstrap_95'],flush=True)
    ax.axhline(.5,color='gray',ls='--'); ax.set(xlabel='Layer',ylabel='Transfer AUC',ylim=(.2,1),title=f'Predicting a later substantive attempt: exit → always_equal\nTrain n={train.sum()}, test n={test.sum()}; rings = training-CV-selected layer')
    ax.legend(); fig.tight_layout()
    Path('out/figs_core').mkdir(exist_ok=True)
    fig.savefig('out/figs_core/fig8_reward_transfer_audit.png',dpi=180)
    (root/'transfer_audit.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
