"""Upload the lens artifacts to the Hub. Private by default.

    HF_TOKEN=hf_... .venv/bin/python upload/push.py            # private
    HF_TOKEN=hf_... .venv/bin/python upload/push.py --public
"""
import argparse, shutil
from pathlib import Path
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="jeeva2812/olmo3-jlens-checkpoints")
ap.add_argument("--public", action="store_true",
                help="off by default: private keeps it reusable without publishing")
a = ap.parse_args()

stage = Path("upload/stage")
if stage.exists():
    shutil.rmtree(stage)
(stage / "lenses").mkdir(parents=True)

shutil.copy("upload/README.md", stage / "README.md")
for f in Path("out/lenses").glob("random_L20_*.pt"):
    shutil.copy(f, stage / "lenses" / f.name)
for f in ["out/lenses/layers_main.pt", "out/Jall_main_fp16.pt"]:
    if Path(f).exists():
        shutil.copy(f, stage / Path(f).name.replace("_fp16", ""))

api = HfApi()
api.create_repo(a.repo, repo_type="model", private=not a.public, exist_ok=True)
api.upload_folder(folder_path=str(stage), repo_id=a.repo, repo_type="model")
print(f"pushed -> https://huggingface.co/{a.repo}  (private={not a.public})")
