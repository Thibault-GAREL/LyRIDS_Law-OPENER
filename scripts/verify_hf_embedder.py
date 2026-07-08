"""Vérifie que l'embedder publié sur le HF Hub est identique au checkpoint local.

Le paper affirme que l'embedder est « loaded from the Hugging Face Hub » alors que
les runs finaux ont utilisé le checkpoint local du papier principal. Ce script
valide (sans GPU) que les deux repos HF portent bit à bit les mêmes poids que le
checkpoint local, ce qui rend la claim du paper exacte.

Usage (venv pytorch_cuda_env, CPU seulement) :
    python -m scripts.verify_hf_embedder
    python -m scripts.verify_hf_embedder --local <chemin> --repos user/repo ...
"""
import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file

LOCAL_DEFAULT = "../LyRIDS_Opener/outputs/models/embedder_contrastive_hard_big"
REPOS_DEFAULT = ["Thibault-GAREL/opener-sup", "Thibault-GAREL/opener-zs"]


def load_state_dict_from_hub(repo_id: str) -> dict:
    """Télécharge et charge les poids d'un repo HF (safetensors ou .bin)."""
    files = set(list_repo_files(repo_id))
    if "model.safetensors" in files:
        path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        return load_file(path)
    if "pytorch_model.bin" in files:
        path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
        return torch.load(path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"{repo_id}: ni model.safetensors ni pytorch_model.bin "
                            f"(fichiers : {sorted(files)})")


def compare(local_sd: dict, remote_sd: dict, tag: str) -> bool:
    """Compare deux state dicts tenseur par tenseur. Retourne True si identiques."""
    local_keys, remote_keys = set(local_sd), set(remote_sd)
    ok = True
    if local_keys != remote_keys:
        ok = False
        missing = sorted(local_keys - remote_keys)[:5]
        extra = sorted(remote_keys - local_keys)[:5]
        print(f"  [{tag}] clés différentes : manquantes={missing} en_trop={extra}")
    n_diff = 0
    worst = 0.0
    common = sorted(local_keys & remote_keys)
    for i, k in enumerate(common, 1):
        a, b = local_sd[k], remote_sd[k]
        if a.shape != b.shape:
            n_diff += 1
            print(f"  [{tag}] {k}: shapes {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        if not torch.equal(a, b):
            n_diff += 1
            worst = max(worst, (a.float() - b.float()).abs().max().item())
        if i % 20 == 0 or i == len(common):
            print(f"  [{tag}] comparé {i}/{len(common)} tenseurs...", flush=True)
    if n_diff:
        ok = False
        print(f"  [{tag}] {n_diff}/{len(common)} tenseurs diffèrent (max |delta| = {worst:.3e})")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", default=LOCAL_DEFAULT)
    parser.add_argument("--repos", nargs="+", default=REPOS_DEFAULT)
    args = parser.parse_args()

    local_path = Path(args.local) / "model.safetensors"
    print(f"Checkpoint local : {local_path}")
    local_sd = load_file(str(local_path))
    print(f"  {len(local_sd)} tenseurs chargés.\n")

    all_ok = True
    for repo in args.repos:
        print(f"Repo HF : {repo}")
        try:
            remote_sd = load_state_dict_from_hub(repo)
        except Exception as exc:
            print(f"  ECHEC téléchargement/chargement : {exc}")
            all_ok = False
            continue
        same = compare(local_sd, remote_sd, repo.split("/")[-1])
        print(f"  => {'IDENTIQUE au checkpoint local' if same else 'DIFFERENT du checkpoint local'}\n")
        all_ok = all_ok and same

    print("VERDICT :", "poids HF == poids locaux, la claim du paper est exacte."
          if all_ok else "au moins un repo diffère, corriger la claim du paper ou republier les poids.")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
