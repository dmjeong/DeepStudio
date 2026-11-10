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
        from libreyolo.models.mobilenetv4.utils import build_eval_transform
        from libreyolo.utils.image_loader import ImageLoader
        from torchvision import transforms

        image = ImageLoader.load(image_path)
        transform = build_eval_transform(int(input_size), self.wrapper.crop_pct)
        stages = transform.transforms
        if not (len(stages) == 4 and isinstance(stages[0], transforms.Resize)
                and isinstance(stages[1], transforms.CenterCrop)):
            raise ValueError("LibreYOLO 전처리가 변경되어 Grad-CAM 좌표를 확인할 수 없습니다")
        resized = stages[0](image)
        tensor = transforms.Compose(stages[1:])(resized).unsqueeze(0)
        activation = self.cam.generate_activation(tensor, target_class=target_class)
        return reproject_classification_cam(
            activation, image.size[::-1], resized.size[::-1], tuple(tensor.shape[-2:]),
            return_valid_mask=True)

    def release(self):
        self.cam.release()
