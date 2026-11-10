"""CAM adapter for the shipped LibreYOLO 1.5 MobileNetV4 classifier."""

import torch

from core.gradcam import GradCAM
from core.heatmap import reproject_classification_cam


def supports_upstream_gradcam(wrapper, task):
    family = str(getattr(wrapper, "FAMILY", getattr(wrapper, "family", ""))).lower()
    model = getattr(wrapper, "model", None)
    return (task == "classify" and family == "mobilenetv4"
            and isinstance(model, torch.nn.Module)
            and isinstance(getattr(model, "blocks", None), torch.nn.Module)
            and isinstance(getattr(model, "classifier", None), torch.nn.Linear))


class LibreYOLOGradCAM:
    def __init__(self, wrapper):
        if not supports_upstream_gradcam(wrapper, "classify"):
            raise ValueError("이 LibreYOLO 모델의 분류 Grad-CAM 연결을 지원하지 않습니다")
        self.wrapper = wrapper
        # conv_head is AFTER global_pool in this architecture: its 1x1 map
        # cannot explain spatial location. blocks is the last spatial map.
        self.cam = GradCAM(wrapper.model, target_layer="blocks", score_layer="classifier")

    def generate_image(self, image_path, input_size, target_class):
        from libreyolo_classification import POLICY
        if getattr(self.wrapper, "studio_input_policy", None) != POLICY:
            raise ValueError("LibreYOLO 전체 영상 전처리가 연결되지 않았습니다")
        tensor, image, _, _ = self.wrapper._preprocess(image_path, input_size=int(input_size))
        activation = self.cam.generate_activation(tensor, target_class=target_class)
        # No implicit crop: all pixels in the selected ROI belong to the CAM.
        return reproject_classification_cam(
            activation, image.size[::-1], tuple(tensor.shape[-2:]), tuple(tensor.shape[-2:]),
            return_valid_mask=True)

    def release(self):
        self.cam.release()
