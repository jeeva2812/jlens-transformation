"""Reconstruct privilege rows from stream logs, so partial runs are usable."""
from __future__ import annotations
import glob, re
from pathlib import Path

ROW = re.compile(
    r"^\s{2}(\w+)\s+(\S+)\s+raw=([-+\d.]+)\s+svd=([-+\d.]+)\s+rep=([-+\d.]+)"
    r"\+-([\d.]+)\s+gram=([-+\d.]+)\s+orth=([-+\d.]+)\s+"
    r"q90=([-+\d.]+)/([-+\d.]+)\s+zq=\s*([-+\d.]+)\s+zrep=\s*([-+\d.]+)"
    r"\s+zgram=\s*([-+\d.]+)\s+dlogdf=([-+\d.]+)")
HDR = re.compile(r"^(\S+): d=\d+ vocab_scored=(\d+) scorer=(\S+)")


def parse(paths=("out/priv/log_*.txt",)):
    rows = []
    for p in sorted({q for g in paths for q in glob.glob(g)}):
        model = scorer = None; vocab = 0
        for ln in Path(p).read_text().splitlines():
            h = HDR.match(ln)
            if h:
                model, scorer, vocab = h.group(1), h.group(3), int(h.group(2))
                continue
            m = ROW.match(ln)
            if not m or model is None:
                continue
            g = m.groups()
            r = {"arm": g[0], "group": g[1], "model": model, "scorer": scorer,
                 "logfile": Path(p).name, "vocab": vocab}
            for i, k in enumerate(["raw", "svd", "reparam_null", "gram_null", "orth_null"]):
                v = float(g[2 + i]) if i < 3 else float(g[3 + i])
                r[k] = {"cos": v, "logdf": 0.0}
            r["reparam_null"]["sd_draw"] = float(g[5])
            r["raw"]["q90"] = float(g[8])
            r["reparam_null"]["q90"] = float(g[9])
            r["zq90_vs_reparam_null"] = float(g[10])
            r["z_vs_reparam_null"] = float(g[11])
            r["z_vs_gram_null"] = float(g[12])
            r["dlogdf"] = float(g[13])
            rows.append(r)
    return rows


if __name__ == "__main__":
    rs = parse()
    from collections import Counter
    print(Counter((r["model"].split("/")[-1], r["scorer"].split("/")[-1]) for r in rs))
    print(f"{len(rs)} rows parsed")
