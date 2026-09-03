import json, torch
ORDER = ["stage1-step0","stage1-step2000","stage1-step8000","stage1-step32000",
         "stage1-step128000","stage1-step512000","stage1-step1413814",
         "stage2-step8000","stage2-step47684","stage3-step5000","main"]
K = 64
layers = torch.load("out/ckpt/J_main.pt", map_location="cpu", weights_only=False)["layers"]
V = {}
for r in ORDER:
    b = torch.load(f"out/ckpt/J_{r}.pt", map_location="cpu", weights_only=False)
    V[r] = {l: torch.linalg.svd(b["J"][l].float())[2][:K].T for l in layers}
    print(f"[svd] {r}", flush=True)
g = torch.Generator().manual_seed(0)
nl = [float(torch.linalg.svdvals(
        torch.linalg.qr(torch.randn(4096, K, generator=g))[0].T @
        torch.linalg.qr(torch.randn(4096, K, generator=g))[0]).mean()) for _ in range(5)]
null = sum(nl)/len(nl)
M = {r: {l: float(torch.linalg.svdvals(V[r][l].T @ V["main"][l]).mean())
         for l in layers} for r in ORDER}
json.dump({"overlap": {r: {str(l): M[r][l] for l in layers} for r in ORDER},
           "null": null, "layers": layers, "order": ORDER},
          open("out/subspace_training.json","w"), indent=2)
print(f"\nchance = {null:.3f}\n")
print(f"{'checkpoint':<22}" + "".join(f"{'L'+str(l):>7}" for l in layers))
for r in ORDER:
    print(f"{r:<22}" + "".join(f"{M[r][l]:>7.2f}" for l in layers))
print("\nWhen does each layer reach 80% of its final subspace?")
for l in layers:
    hit = next((r for r in ORDER if M[r][l] >= .80), "never")
    print(f"  L{l:<3} {hit:<24} (end of pretraining: {M['stage1-step1413814'][l]:.2f})")
