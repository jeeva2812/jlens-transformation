"""One label-free interpretability score, shared by every experiment here.

An earlier count of "interpretable directions" was scored against a hand-written
list of ~35 concepts. That count is bounded by what I happened to write down: a
direction meaning something not on the list scores zero. So instead ask a
concept-free question -- are this direction's top tokens close together?

Two versions, because the naive one is gameable:

  coh_z  mean pairwise cosine of the top tokens' unembedding rows, z-scored
         against token sets of the SAME rarity. The rarity match matters: a
         random direction in a 49k vocab picks out rare tokens, and rare tokens
         cluster with each other for reasons that have nothing to do with
         meaning. Without the match, random directions outscore real ones.

  xm_z   the same tokens, scored in a DIFFERENT model's embedding space.
         Trained separately, different data. A cluster that survives the move
         is about the words, not about this model's tokenizer.

Both are z-scores: how many standard deviations above a matched null.
"""
from __future__ import annotations
import statistics as st
import torch, torch.nn.functional as F


class Scorer:
    def __init__(self, W_U, tok, other_model=None, nnull=200, seed=0):
        Wc = W_U - W_U.mean(0, keepdim=True)   # the mean direction is the
        self.Wn = F.normalize(Wc, dim=1)       # attention sink, not a concept
        self.Wc = Wc
        nrm = Wc.norm(dim=1)
        self.order = torch.argsort(nrm)
        self.rank = torch.empty_like(self.order)
        self.rank[self.order] = torch.arange(len(self.order))
        self.g = torch.Generator().manual_seed(seed)
        self.nnull = nnull
        self.OE = self.xmap = None
        if other_model:
            self._load_other(other_model, tok)

    def _load_other(self, name, tok):
        from transformers import AutoModel, AutoTokenizer
        otok = AutoTokenizer.from_pretrained(name)
        E = AutoModel.from_pretrained(name, dtype=torch.float32).get_input_embeddings()
        OE = E.weight.detach().float()
        self.OE = F.normalize(OE - OE.mean(0, keepdim=True), dim=1)
        n = self.Wn.shape[0]
        m = torch.full((n,), -1, dtype=torch.long)
        vocab = otok.get_vocab()
        for i, s in enumerate(tok.convert_ids_to_tokens(list(range(n)))):
            j = vocab.get(s)
            if j is not None and j < self.OE.shape[0]:
                m[i] = j
        self.xmap = m
        self.shared = int((m >= 0).sum())

    @staticmethod
    def _pair(V):
        C = V @ V.T; n = V.shape[0]
        return float((C.sum() - n) / (n * (n - 1)))

    def coh_z(self, ids):
        """tighter than tokens of the same rarity?"""
        real = self._pair(self.Wn[ids])
        null = []
        for _ in range(self.nnull):
            pick = [int(self.order[torch.randint(max(0, int(r) - 500),
                                                 min(len(self.order), int(r) + 500),
                                                 (1,), generator=self.g)])
                    for r in self.rank[ids]]
            null.append(self._pair(self.Wn[torch.tensor(pick)]))
        return (real - st.mean(null)) / (st.pstdev(null) + 1e-9)

    def xm_z(self, ids):
        """still tight in a model trained separately on different data?"""
        if self.OE is None: return float("nan")
        j = self.xmap[ids]; j = j[j >= 0]
        if len(j) < 6: return float("nan")
        real = self._pair(self.OE[j])
        null = [self._pair(self.OE[torch.randint(0, self.OE.shape[0], (len(j),),
                                                 generator=self.g)])
                for _ in range(self.nnull)]
        return (real - st.mean(null)) / (st.pstdev(null) + 1e-9)

    def topk(self, direction, k=12):
        return torch.topk(self.Wc @ direction, k).indices

    def n_shared(self, ids):
        """how many of these tokens exist at all in the other model's vocabulary --
        a direction pointing at tokens no other tokenizer has is a tokenizer
        artefact, and that is worth counting rather than discarding"""
        return 0 if self.xmap is None else int((self.xmap[ids] >= 0).sum())

    @staticmethod
    def is_content(toks):
        """Does this direction point at words, or at typography?

        Quote marks and brackets cluster tightly in every model's embedding, so
        they saturate the cross-model score while meaning nothing -- the top of
        the "most interpretable" list came back as nothing but curly quotes.
        A direction counts as content only if most of its top tokens are real
        word-pieces: three or more letters, no punctuation glued on.
        """
        def word(t):
            t = t.strip()
            return len(t) >= 3 and sum(c.isalpha() for c in t) >= 3 \
                   and all(c.isalpha() or c in "-'" for c in t)
        # two thirds, not half: at half, directions whose top tokens are all
        # newlines still slipped through on the strength of their tail
        return sum(word(t) for t in toks) >= (2 * len(toks)) // 3

    def score(self, direction, k=12):
        ids = self.topk(direction, k)
        return dict(coh_z=self.coh_z(ids), xm_z=self.xm_z(ids),
                    n_shared=self.n_shared(ids), ids=ids)
