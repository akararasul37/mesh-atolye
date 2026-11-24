import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
ENGINE = RUNTIME / "hunyuan"
MODELS = ROOT / "models"
MV_MODEL = MODELS / "hunyuan-mv" / "hunyuan3d-dit-v2-mv"
FRONTEND = ROOT / "frontend"
JOBS = ROOT / "uretimler"
INSTALL = RUNTIME / "installed.json"


def multiview_ready() -> bool:
    weights = MV_MODEL / "model.fp16.ckpt"
    return (MV_MODEL / "config.yaml").is_file() and weights.is_file() and weights.stat().st_size == 4928618068


def capabilities() -> dict:
    try:
        data = json.loads(INSTALL.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {"geometry": False, "pbr": False}
    return {**data, "multiview": multiview_ready()}
