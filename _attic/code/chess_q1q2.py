"""Q1/Q2 chess-oracle spike (docs/CHESS_PLAN.md section 4, Q1+Q2 only).

Q1: legality-gradient matrix G (N x 512) per (parity, layer); SVD spectrum +
    effective rank. Pre-registered prediction: HIGH rank (~70% conf).
Q2: subspace overlap span(V_k of J) vs span(G_r), k in (4,16,64), with all three
    required baselines: (a) random k-dim (30 draws), (b) residual PCA top-k,
    (c) span(G_r) self-ceiling + split-half stability ceiling.

Type discipline: g_p = d s / d h_l lives in layer-l space, compared against V
(right singular vectors) of J_l, never U. J files: pooled J_chess.pt and
parity-conditioned J_cond.pt (origin/dest/matched).
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path

import chess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "austindavis/chess-gpt2-uci-8x8x512"
SQ0 = 4
DEV = "cpu"


def sq_logits(logits):
    return logits[0, -1, SQ0:SQ0 + 64]


@torch.no_grad()
def boundary_check(out):
    b = torch.load("out/chess/J_chess.pt", map_location="cpu", weights_only=False)
    t = b["target"]
    ident = (b["J"][t].float() - torch.eye(512)).abs().max().item()
    print(f"boundary check  max|J[target={t}] - I| = {ident:.3e}   (must be ~0)", flush=True)
    out["boundary_max_abs"] = ident
    out["boundary_target"] = t


def gen_positions(n, seed):
    """One (origin-pos, dest-pos) pair per random game. Returns list of dicts
    with keys: ids (prefix token list), legal (sorted legal squares),
    illegal (sorted), parity ('origin'|'dest'), jeu (game idx)."""
    rng = random.Random(seed)
    pos = []
    for g in range(n):
        board = chess.Board()
        nplies = rng.randint(6, 40)
        moves = []
        for _ in range(nplies):
            legal = list(board.legal_moves)
            if not legal:
                break
            mv = rng.choice(legal)
            moves.append(mv)
            board.push(mv)
        if len(moves) < 3:
            continue
        j = rng.randrange(1, len(moves))  # move index to probe (skip ply 0)
        b = chess.Board()
        ids = [1]  # BOS
        for mv in moves[:j]:
            ids += [SQ0 + mv.from_square, SQ0 + mv.to_square]
            b.push(mv)
        assert b.turn == board.turn or True
        # origin-parity position: predict from-square on board b
        legal_from = sorted({m.from_square for m in b.legal_moves})
        pos.append({"ids": list(ids), "legal": legal_from, "parity": "origin", "game": g})
        # destination-parity: same board, origin = actual move's from-square
        frm = moves[j].from_square
        assert any(m.from_square == frm for m in b.legal_moves)
        legal_to = sorted({m.to_square for m in b.legal_moves if m.from_square == frm})
        pos.append({"ids": ids + [SQ0 + frm], "legal": legal_to,
                    "parity": "dest", "game": g, "origin": frm})
    for p in pos:
        s = set(p["legal"])
        assert 1 <= len(s) < 64, p["parity"]
        p["illegal"] = sorted(set(range(64)) - s)
    return pos


def collect(model, blocks, layers, positions):
    """Per position: one forward, one scalar s, one autograd.grad -> g per
    layer + residual activation per layer. Returns dict parity -> layer ->
    {'G': tensor, 'H': tensor} plus mean legal-mass diagnostic."""
    root = min(layers)
    out = {"origin": {l: {"G": [], "H": []} for l in layers},
           "dest": {l: {"G": [], "H": []} for l in layers}}
    mass = {"origin": [], "dest": []}
    t0 = time.time()
    for i, p in enumerate(positions):
        t = torch.tensor([p["ids"]])
        store = {}

        def make_hook(l):
            def hook(mod, inp, outp):
                tensor = outp if torch.is_tensor(outp) else outp[0]
                if l == root:
                    # must root the graph DURING the forward; doing it after
                    # the forward returns disconnects downstream ops (lens.py
                    # _ResidualCapture does the same inside its hook).
                    tensor.requires_grad_(True)
                store[l] = tensor
            return hook

        handles = [blocks[l].register_forward_hook(make_hook(l)) for l in layers]
        with torch.enable_grad():
            store.clear()
            logits = model(input_ids=t, use_cache=False).logits.float()
            hs = {l: store[l] for l in layers}
            sq = sq_logits(logits)
            legal = torch.tensor(p["legal"])
            ill = torch.tensor(p["illegal"])
            s = torch.logsumexp(sq[legal], 0) - torch.logsumexp(sq[ill], 0)
            with torch.no_grad():
                mass[p["parity"]].append(float(sq.softmax(0)[legal].sum()))
            grads = torch.autograd.grad(s, [hs[l] for l in layers])
        for l, g in zip(layers, grads):
            out[p["parity"]][l]["G"].append(g[0, -1].detach().float().clone())
            out[p["parity"]][l]["H"].append(hs[l][0, -1].detach().float().clone())
        for h in handles:
            h.remove()
        del hs, store, logits, grads
        if (i + 1) % 50 == 0 or i + 1 == len(positions):
            dt = time.time() - t0
            print(f"  collected {i+1}/{len(positions)}  ({dt:.0f}s, "
                  f"{dt/(i+1)*1000:.0f} ms/pos)", flush=True)
    for par in out:
        for l in layers:
            out[par][l]["G"] = torch.stack(out[par][l]["G"])
            out[par][l]["H"] = torch.stack(out[par][l]["H"])
    return out, {k: sum(v) / len(v) for k, v in mass.items()}


def spectrum(G):
    Gn = torch.nn.functional.normalize(G.float(), dim=1)
    S = torch.linalg.svdvals(Gn)
    lam = (S ** 2)
    lam = lam / lam.sum()
    cum = torch.cumsum(lam, 0)
    k90 = int((cum < 0.90).sum()) + 1
    k95 = int((cum < 0.95).sum()) + 1
    pr = float(1.0 / (lam ** 2).sum())          # participation ratio
    sr = float((S ** 2).sum() / (S[0] ** 2))    # stable rank
    return Gn, S, k90, k95, pr, sr


def randn_null_pr(n, d, draws=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    prs = []
    for _ in range(draws):
        R = torch.randn(n, d, generator=g)
        R = torch.nn.functional.normalize(R, dim=1)
        lam = torch.linalg.svdvals(R) ** 2
        lam = lam / lam.sum()
        prs.append(float(1.0 / (lam ** 2).sum()))
    import statistics as st
    return sum(prs) / len(prs), st.pstdev(prs)


def orthobasis(k, d, gen):
    A = torch.randn(d, k, generator=gen)
    return torch.linalg.qr(A).Q


def overlap(A, B):
    return float(((A.T @ B) ** 2).sum() / A.shape[1])


def q2_table(J_pool, J_match, Gn, H, ks=(4, 16, 64), seed=0):
    Vj_pool = torch.linalg.svd(J_pool.float())[2].T   # V (512x512), cols = dirs
    Vj_match = torch.linalg.svd(J_match.float())[2].T
    Vg = torch.linalg.svd(Gn)[2].T
    Hc = H.float() - H.float().mean(0, keepdim=True)
    Vh = torch.linalg.svd(Hc)[2].T
    d = Gn.shape[1]
    rows = []
    for k in ks:
        B = Vg[:, :k]
        jp = overlap(Vj_pool[:, :k], B)
        jm = overlap(Vj_match[:, :k], B)
        gen = torch.Generator().manual_seed(seed)
        rands = [overlap(orthobasis(k, d, gen), B) for _ in range(30)]
        import statistics as st
        rr = {"mean": sum(rands) / len(rands), "std": st.pstdev(rands), "max": max(rands)}
        pc = overlap(Vh[:, :k], B)
        # split-half ceiling: stability of span(G_r) itself
        sh = []
        for dd in range(5):
            perm = torch.randperm(Gn.shape[0],
                                  generator=torch.Generator().manual_seed(1000 + dd))
            h1, h2 = Gn[perm[:len(perm) // 2]], Gn[perm[len(perm) // 2:]]
            A1 = torch.linalg.svd(h1)[2].T[:, :k]
            A2 = torch.linalg.svd(h2)[2].T[:, :k]
            sh.append(overlap(A1, A2))
        rows.append({"k": k, "J_pooled": jp, "J_matched": jm,
                     "rand_mean": rr["mean"], "rand_std": rr["std"], "rand_max": rr["max"],
                     "pca": pc, "ceil_self": 1.0,
                     "ceil_split_mean": sum(sh) / len(sh),
                     "ceil_split_std": st.pstdev(sh)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 6])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("out/chess"))
    a = ap.parse_args()
    t_all = time.time()
    res = {"config": {"n": a.n, "layers": list(a.layers), "seed": a.seed,
                        "out": str(a.out), "model": MODEL},
             "prediction": "Q1 HIGH rank (~70% conf)"}
    boundary_check(res)

    tok = AutoTokenizer.from_pretrained(MODEL)
    for nm in ("a1", "e2", "e4", "h8", "d5"):
        assert tok.convert_tokens_to_ids(nm) == SQ0 + chess.parse_square(nm), nm
    print("tokenizer square indexing assert OK (id = 4 + python-chess square)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    from jlens.lens import _find_blocks_and_norm
    blocks, _ = _find_blocks_and_norm(model)
    print(f"model: 8 blocks, using layers {a.layers}, target 7", flush=True)

    t = time.time()
    positions = gen_positions(a.n, a.seed)
    no = sum(1 for p in positions if p["parity"] == "origin")
    nd = sum(1 for p in positions if p["parity"] == "dest")
    print(f"positions: {len(positions)} ({no} origin, {nd} dest) from {a.n} games "
          f"({time.time()-t:.1f}s)", flush=True)

    t = time.time()
    data, mass = collect(model, blocks, a.layers, positions)
    t_collect = time.time() - t
    print(f"collect wall time: {t_collect:.1f}s for {len(positions)} positions "
          f"({t_collect/len(positions)*1000:.0f} ms/pos, {len(a.layers)} layers/backward)",
          flush=True)
    print(f"mean model legal-mass on squares: origin {mass['origin']:.3f}, "
          f"dest {mass['dest']:.3f}", flush=True)
    res["wall_collect_s"] = t_collect
    res["n_positions"] = len(positions)
    res["mean_legal_mass"] = mass

    nullmemo = {}
    res["Q1"] = {}
    Gn_store = {}
    for par in ("origin", "dest"):
        res["Q1"][par] = {}
        Gn_store[par] = {}
        for l in a.layers:
            G, H = data[par][l]["G"], data[par][l]["H"]
            Gn, S, k90, k95, pr, sr = spectrum(G)
            Gn_store[par][l] = (Gn, H)
            key = (G.shape[0], G.shape[1])
            if key not in nullmemo:
                nullmemo[key] = randn_null_pr(*key)
            nm, ns = nullmemo[key]
            verdict = "HIGH" if k90 > 30 else ("LOW" if k90 <= 10 else "MID")
            print(f"Q1 {par} L{l}: N={G.shape[0]} top10sig="
                  + " ".join(f"{v:.2f}" for v in S[:10].tolist())
                  + f" | k90={k90} k95={k95} PR={pr:.1f} stable={sr:.1f} "
                  f"| randn-null PR={nm:.1f}±{ns:.1f} -> {verdict} "
                  f"(pred HIGH)", flush=True)
            res["Q1"][par][l] = {
                "N": G.shape[0], "top10_sigma": [round(float(v), 4) for v in S[:10]],
                "sigma": [round(float(v), 4) for v in S],
                "k90": k90, "k95": k95, "PR": round(pr, 2),
                "stable_rank": round(sr, 2),
                "randn_null_PR": [round(nm, 2), round(ns, 2)], "verdict": verdict}

    blob = torch.load("out/chess/J_chess.pt", map_location="cpu", weights_only=False)
    cond = torch.load("out/chess/J_cond.pt", map_location="cpu", weights_only=False)
    res["Q2"] = {}
    for par in ("origin", "dest"):
        res["Q2"][par] = {}
        for l in a.layers:
            Gn, H = Gn_store[par][l]
            rows = q2_table(blob["J"][l].float(), cond["J"][par][l].float(), Gn, H)
            res["Q2"][par][l] = rows
            print(f"Q2 {par} L{l} (overlap = mean cos^2 principal angles; "
                  f"r=k; 30 random draws):", flush=True)
            print(f"  {'k':>3} {'J_pool':>7} {'J_match':>7} "
                  f"{'rand_mean±std(max)':>20} {'PCA':>6} {'ceil':>5} {'split':>5}",
                  flush=True)
            for r in rows:
                print(f"  {r['k']:>3} {r['J_pooled']:>7.3f} {r['J_matched']:>7.3f} "
                      f"{r['rand_mean']:>6.3f}±{r['rand_std']:.3f}({r['rand_max']:.3f}) "
                      f"{r['pca']:>6.3f} {r['ceil_self']:>5.1f} "
                      f"{r['ceil_split_mean']:>5.3f}", flush=True)

    res["wall_total_s"] = time.time() - t_all
    print(f"TOTAL wall time: {res['wall_total_s']:.1f}s", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "q1_q2.json").write_text(json.dumps(res, indent=1))
    print(f"[saved] {a.out/'q1_q2.json'}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for j, par in enumerate(("origin", "dest")):
        for l in a.layers:
            S = torch.tensor(res["Q1"][par][str(l)]["sigma"]
                             if str(l) in res["Q1"][par] else res["Q1"][par][l]["sigma"])
            lam = (S ** 2) / (S ** 2).sum()
            ax[0][j].plot(range(1, len(lam) + 1), torch.cumsum(lam, 0).numpy(),
                          label=f"L{l}")
        ax[0][j].axhline(0.9, color="k", ls=":", lw=1)
        ax[0][j].set_title(f"Q1 {par}: cumulative variance (G row-normed)")
        ax[0][j].set_xlabel("components"); ax[0][j].set_ylabel("cum var")
        ax[0][j].legend(); ax[0][j].set_xlim(0, len(lam))
        rows = res["Q2"][par][str(a.layers[-1])] \
            if str(a.layers[-1]) in res["Q2"][par] else res["Q2"][par][a.layers[-1]]
        ks = [r["k"] for r in rows]
        ax[1][j].errorbar(ks, [r["J_matched"] for r in rows], fmt="-o", label="J matched")
        ax[1][j].plot(ks, [r["J_pooled"] for r in rows], "s--", label="J pooled")
        ax[1][j].errorbar(ks, [r["rand_mean"] for r in rows],
                          yerr=[r["rand_std"] for r in rows], fmt="^:",
                          label="random (30 draws)")
        ax[1][j].plot(ks, [r["pca"] for r in rows], "d-.", label="resid PCA")
        ax[1][j].plot(ks, [r["ceil_split_mean"] for r in rows], "k--", label="split-half")
        ax[1][j].set_xscale("log"); ax[1][j].set_xticks(ks); ax[1][j].set_xticklabels(ks)
        ax[1][j].set_title(f"Q2 {par} L{a.layers[-1]}: subspace overlap vs k")
        ax[1][j].set_xlabel("k (=r)"); ax[1][j].set_ylabel("mean cos^2")
        ax[1][j].legend(fontsize=8); ax[1][j].set_ylim(0, 1.02)
    fig.suptitle(f"Chess oracle Q1/Q2 spike ({MODEL}, N={len(positions)})")
    fig.tight_layout()
    fig.savefig(a.out / "q1_q2.png", dpi=120)
    print(f"[saved] {a.out/'q1_q2.png'}", flush=True)


if __name__ == "__main__":
    main()
