#!/usr/bin/env python3
"""Upload the Day22 DPO adapter and reviewed model card to a public HF repo.

Set HF_TOKEN in the repo-local .env first. Optionally set HF_REPO to an
existing/new repository under the authenticated account.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lab22 import config as C  # loads the repo-local .env without printing it


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is missing from D:\\Day22\\.env")
    api = HfApi(token=token)
    username = api.whoami()["name"]
    repo_id = os.environ.get("HF_REPO") or f"{username}/lab22-dpo-vn-long-2A202602471"
    if repo_id.split("/", 1)[0].lower() != username.lower():
        raise SystemExit("HF_REPO must belong to the authenticated account")

    adapter = C.DPO_ADAPTER
    files = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
    )
    for name in files:
        if not (adapter / name).is_file():
            raise SystemExit(f"Missing adapter file: {name}")
    card = C.REPO_ROOT / "submission" / "MODEL_CARD.md"
    if not card.is_file():
        raise SystemExit("Missing submission/MODEL_CARD.md")

    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    for name in files:
        api.upload_file(
            path_or_fileobj=str(adapter / name),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add Day22 DPO {name}",
        )
        print(f"Uploaded {name}", flush=True)
    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add evaluated model card",
    )
    print(f"HF_MODEL_URL=https://huggingface.co/{repo_id}", flush=True)


if __name__ == "__main__":
    main()
