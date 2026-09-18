"""Re-DETR and SAM2 multi-graph manifest contracts."""

import pytest

from model_runtime.special_contracts import (
    SpecialContractError,
    validate_re_detr_manifest,
    validate_sam2_manifest,
)


def _redetr():
    return {
        "family": "Re-DETR v4", "variant": "Medium", "task": "detect",
        "runtimes": ["container", "onnx"],
        "contracts": {"onnx": {"input_name": "input_image", "boxes_name": "pred_boxes",
                                 "logits_name": "pred_logits", "boxes_format": "normalized_cxcywh",
                                 "score_activation": "sigmoid"}},
    }


def _sam2():
    return {
        "family": "SAM2", "variant": "Hiera Small", "task": "segment",
        "runtimes": ["container", "onnx"],
        "contracts": {"graphs": {
            "encoder": {"file": "sam2_encoder.onnx", "outputs": ["image_embeddings"]},
            "decoder": {"file": "sam2_decoder.onnx", "outputs": ["low_res_mask_logits", "iou_predictions"]},
        }, "prompt_types": ["point", "box", "mask"], "video_state": True},
    }


def test_special_catalog_contracts_accept_all_requested_variants():
    validate_re_detr_manifest(_redetr())
    validate_sam2_manifest(_sam2())
    for variant in ("Small", "Medium", "Large"):
        item = _redetr(); item["variant"] = variant
        validate_re_detr_manifest(item)
    for variant in ("Hiera Tiny", "Hiera Small", "Hiera Base+", "Hiera Large"):
        item = _sam2(); item["variant"] = variant
        validate_sam2_manifest(item)


@pytest.mark.parametrize("mutator", [
    lambda item: item["contracts"]["onnx"].pop("boxes_name"),
    lambda item: item.update(variant="XL"),
])
def test_redetr_contract_rejects_incomplete_manifest(mutator):
    item = _redetr()
    mutator(item)
    with pytest.raises(SpecialContractError):
        validate_re_detr_manifest(item)


@pytest.mark.parametrize("mutator", [
    lambda item: item["contracts"]["graphs"].pop("decoder"),
    lambda item: item["contracts"].update(video_state="yes"),
])
def test_sam2_contract_rejects_incomplete_manifest(mutator):
    item = _sam2()
    mutator(item)
    with pytest.raises(SpecialContractError):
        validate_sam2_manifest(item)
