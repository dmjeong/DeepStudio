"""Actionable dependency checks shared by training, checkpoint load and export."""

import importlib
import sys
from pathlib import Path

TIMM_MODELS = frozenset({
    "unet_mobilenetv4_small", "unet_efficientnet_b1",
    "efficientvit_b0_seg", "efficientvit_b1_seg", "eomt_dinov2_small",
})


def dependency_message(model_id, error):
    missing = getattr(error, "name", None) or str(error)
    explanation = (f"{model_id}: 모델 실행 패키지({missing})를 불러오지 못했습니다. "
                   "이미지나 가중치 문제가 아니며 재학습할 필요가 없습니다.\n")
    if getattr(sys, "frozen", False):
        return explanation + "EXE에 필요한 패키지가 누락됐습니다. 수정된 배포판으로 업데이트하세요.\n" + str(error)
    script = Path(__file__).resolve().parents[1] / "gui" / "repair_model_runtime.py"
    return (explanation + "앱을 종료한 뒤 gui 폴더의 repair_model_runtime.bat을 실행하세요.\n"
            "기존 GPU용 PyTorch와 사전학습 가중치를 다시 받지 않습니다.\n"
            f"현재 실행 Python: {sys.executable}\n"
            f'같은 Python으로 실행할 명령: "{sys.executable}" "{script}"\n'
            f"원래 오류: {error}")


def require_model_runtime(model_id):
    modules = ["timm.layers", "timm.models.efficientvit_mit"] if model_id in TIMM_MODELS else []
    if model_id == "eomt_dinov2_small":
        modules.append("scipy.optimize")
    for module in modules:
        try:
            importlib.import_module(module)
        except (ImportError, OSError) as exc:
            raise RuntimeError(dependency_message(model_id, exc)) from exc
