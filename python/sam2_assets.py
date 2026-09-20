"""Official SAM2.1 pretrained-asset discovery, download, and model loading.

The Studio deliberately keeps the large checkpoint files out of source control.
They are downloaded from Meta's official Hugging Face repositories into a user
selected local cache, verified by the hub client, and then loaded by the
official SAM-2 package. This is the mapping shared by Settings, native
inference, and future fine-tuning/export adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Sam2Asset:
    model_id: str
    variant: str
    hub_model_id: str
    filename: str
    config_name: str


SAM2_ASSETS: dict[str, Sam2Asset] = {
    "sam2_hiera_tiny": Sam2Asset("sam2_hiera_tiny", "Hiera Tiny", "facebook/sam2.1-hiera-tiny", "sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "sam2_hiera_small": Sam2Asset("sam2_hiera_small", "Hiera Small", "facebook/sam2.1-hiera-small", "sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "sam2_hiera_base_plus": Sam2Asset("sam2_hiera_base_plus", "Hiera Base+", "facebook/sam2.1-hiera-base-plus", "sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "sam2_hiera_large": Sam2Asset("sam2_hiera_large", "Hiera Large", "facebook/sam2.1-hiera-large", "sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}


def get_sam2_asset(model_id: str) -> Sam2Asset:
    try:
        return SAM2_ASSETS[model_id]
    except KeyError as exc:
        raise ValueError(f"지원하지 않는 SAM2 모델 ID: {model_id}") from exc


def download_sam2_pretrained(model_id: str, cache_dir: str | Path | None = None) -> Path:
    """Download exactly one official checkpoint through Hugging Face Hub."""
    asset = get_sam2_asset(model_id)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("SAM2 사전학습 가중치 다운로드 구성요소가 없습니다. Deep Vision Studio 설치본을 다시 빌드하세요.") from exc
    kwargs: dict[str, Any] = {"repo_id": asset.hub_model_id, "filename": asset.filename}
    if cache_dir is not None:
        target = Path(cache_dir).expanduser()
        if not target.is_absolute():
            raise ValueError("SAM2 가중치 저장 위치는 절대 경로여야 합니다.")
        target.mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = str(target)
    try:
        path = Path(hf_hub_download(**kwargs))
    except Exception as exc:
        raise RuntimeError(f"SAM2.1 {asset.variant} 공식 가중치 다운로드 실패: {exc}") from exc
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("SAM2.1 가중치 다운로드가 완료되지 않았습니다.")
    return path.resolve()


def load_sam2_pretrained(model_id: str, *, device: str = "cpu", checkpoint_path: str | Path | None = None):
    """Build an official SAM2.1 image model from a downloaded checkpoint."""
    asset = get_sam2_asset(model_id)
    path = (Path(checkpoint_path).expanduser() if checkpoint_path is not None
            else download_sam2_pretrained(model_id))
    if not path.is_file():
        raise FileNotFoundError(f"SAM2 가중치 파일이 없습니다: {path}")
    try:
        from sam2.build_sam import build_sam2
    except ImportError as exc:
        raise RuntimeError("SAM2 공식 런타임이 없습니다. build.bat으로 설치본을 다시 빌드하세요.") from exc
    try:
        return build_sam2(asset.config_name, ckpt_path=str(path), device=device, mode="eval")
    except Exception as exc:
        raise RuntimeError(f"SAM2.1 {asset.variant} 로드 실패: {exc}") from exc
