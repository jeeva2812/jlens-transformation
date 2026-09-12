"""Rebuild every application figure that only needs checked-in JSON results.

No model weights or inference are required. Figure 8 is intentionally omitted:
its producer compares full Jacobian tensors, which are too large to check in.

Run from the repository root:

    MPLCONFIGDIR=/tmp/jlens-mpl PYTHONPATH=. \
      ./.venv/bin/python -m jlens.rebuild_figures
"""
from __future__ import annotations

from jlens import final_figs, layer_subspaces_traj_fig, training_fig
from jlens import training_spectrum_fig


def main() -> None:
    final_figs.OUT.mkdir(parents=True, exist_ok=True)
    for name, build in (
        ("fig1", final_figs.fig1),
        ("fig2", final_figs.fig2),
        ("fig3", final_figs.fig3),
        ("fig4", final_figs.fig4),
        ("fig5", final_figs.fig5),
    ):
        build()
        print(f"rebuilt {name}")

    training_fig.main()
    training_spectrum_fig.main()
    layer_subspaces_traj_fig.main()
    print("\nrebuilt application figures 1–7 and 9 in out/figs_final/")
    print("figure 8 needs the full Jacobian tensors; the checked-in PNG is retained")


if __name__ == "__main__":
    main()
