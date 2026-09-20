"""OpenCV와 ONNX Runtime만 사용하는 EfficientNet CPU 분류기."""
import json
from pathlib import Path
import time

import numpy as np


def validate_config(config):
    from opencv_preprocess import resize_contract
    from center_crop import validate_center_crop
    if (type(config.get("schema_version")) is not int or config["schema_version"] not in (5, 6) or
            config.get("backend") != "efficientnet" or config.get("task") != "classify"):
        raise ValueError("현재 버전에서 내보낸 EfficientNet JSON 필요")
    from onnx_session import deployment_session_settings
    deployment_session_settings(config)
    for key in ("input_height", "input_width", "input_channels", "num_classes"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"양의 정수 필요: {key}")
    channels = config["input_channels"]
    if channels not in (1, 3) or config.get("postprocessing", {}).get("output") != "logits":
        raise ValueError("EfficientNet 입력 채널 또는 logits 계약 오류")
    prep = config.get("preprocessing", {})
    if prep.get("resize_implementation") != "opencv_linear_exact_v1":
        raise ValueError("OpenCV 전처리로 다시 내보내세요")
    resize_contract(prep)
    if prep.get("color_order") != ("GRAY" if channels == 1 else "RGB"):
        raise ValueError("모델과 전처리 색상 순서 불일치")
    if "in_channels" in prep and (type(prep["in_channels"]) is not int or prep["in_channels"] != channels):
        raise ValueError("중복 저장된 입력 채널 불일치")
    if config.get("model_config") is not None:
        from efficientnet_contract import model_input_contract
        contract = model_input_contract(config["model_config"], channels)
        if prep.get("grayscale_adapter", contract["input_adapter"]) != contract["input_adapter"]:
            raise ValueError("모델과 전처리 입력 변환 방식 불일치")
    validate_center_crop(prep.get("center_crop"))
    for key in ("normalize_mean", "normalize_std"):
        values = np.asarray(config.get(key, []), dtype=np.float32)
        if values.shape != (channels,) or not np.isfinite(values).all() or (key.endswith("std") and np.any(values <= 0)):
            raise ValueError("정규화 계수 오류")
        if key in prep and not np.array_equal(values, np.asarray(prep[key], np.float32)):
            raise ValueError("중복 저장된 정규화 계수 불일치")
    names = config.get("class_names", [])
    if names and (len(names) != config["num_classes"] or not all(isinstance(n, str) for n in names)):
        raise ValueError("클래스 목록 오류")
    return config


class ImageClassifier:
    """공유 전처리/후처리. 런타임은 logits(tensor)를 구현한다."""

    def __init__(self, config_path):
        self.config = validate_config(json.loads(Path(config_path).read_text(encoding="utf-8")))
        cfg = self.config
        self.mean = np.asarray(cfg["normalize_mean"], np.float32)[:, None, None]
        self.std = np.asarray(cfg["normalize_std"], np.float32)[:, None, None]
        from efficientnet_contract import model_input_contract, LEGACY_GRAY_INPUT
        definition = cfg.get("model_config")
        self.input_contract = model_input_contract(definition, cfg["input_channels"]) if definition is not None else None
        if cfg["preprocessing"].get("grayscale_adapter") == LEGACY_GRAY_INPUT or (
                self.input_contract is not None and self.input_contract["input_adapter"] == LEGACY_GRAY_INPUT):
            import warnings
            warnings.warn("구형 EfficientNet: 외부 입력은 1ch이며 모델 내부 첫 Conv는 3ch입니다.", UserWarning, stacklevel=2)

    def preprocess(self, rgb):
        """GRAY [H,W]/[H,W,1] 또는 RGB/RGBA uint8 배열을 모델 입력으로 변환한다."""
        from center_crop import center_crop_box
        from opencv_preprocess import convert_channels, resize
        cfg = self.config
        rgb = np.asarray(rgb)
        if rgb.ndim not in (2, 3) or rgb.dtype != np.uint8 or not rgb.size:
            raise ValueError("비어 있지 않은 uint8 이미지 필요")
        box = center_crop_box((rgb.shape[1], rgb.shape[0]), cfg["preprocessing"].get("center_crop"))
        pixels = convert_channels(rgb[box[1]:box[3], box[0]:box[2]], cfg["input_channels"])
        pixels = resize(pixels, (cfg["input_height"], cfg["input_width"]))
        chw = pixels[None] if pixels.ndim == 2 else pixels.transpose(2, 0, 1)
        return np.ascontiguousarray(((chw.astype(np.float32) / np.float32(255) - self.mean) / self.std)[None])

    def logits(self, tensor):
        raise NotImplementedError

    def predict_rgb(self, rgb):
        started = time.perf_counter()
        tensor = self.preprocess(rgb)
        prepared = time.perf_counter()
        logits = self.logits(tensor)[0]
        inferred = time.perf_counter()
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        # FP32 softmax can round close logits to equal probabilities. Preserve rank.
        index = int(logits.argmax())
        names = self.config.get("class_names") or [str(i) for i in range(len(probabilities))]
        confidence = float(probabilities[index])
        probability_list = probabilities.tolist()
        finished = time.perf_counter()
        return {"class_id": index, "class_name": names[index], "confidence": confidence,
                "probabilities": probability_list,
                "preprocess_ms": (prepared-started)*1000, "model_ms": (inferred-prepared)*1000,
                "postprocess_ms": (finished-inferred)*1000, "total_ms": (finished-started)*1000}

    def predict_gray(self, gray):
        """외부 입력이 1채널인 모델에 흑백 카메라 배열을 직접 전달한다."""
        gray = np.asarray(gray)
        if self.config["input_channels"] != 1 or not (
                gray.ndim == 2 or (gray.ndim == 3 and gray.shape[2] == 1)):
            raise ValueError("predict_gray에는 1채널 모델과 흑백 이미지 필요")
        return self.predict_rgb(gray)

    def predict_file(self, path):
        from opencv_preprocess import read_image
        started = time.perf_counter()
        pixels = read_image(path, self.config["input_channels"])
        decoded = time.perf_counter()
        result = self.predict_rgb(pixels)
        result["decode_ms"] = (decoded-started)*1000
        result["file_total_ms"] = (time.perf_counter()-started)*1000
        return result


class OnnxClassifier(ImageClassifier):
    def __init__(self, config_path, *, num_threads=None):
        import onnxruntime as ort
        from onnx_session import cpu_session_options, deployment_session_settings
        path = Path(config_path)
        super().__init__(config_path)
        if num_threads is None:
            num_threads = self.config.get("num_threads", 0)
        if type(num_threads) is not int or num_threads < 0:
            raise ValueError("스레드 수는 0 이상의 정수 필요")
        cfg = self.config
        settings = deployment_session_settings(cfg)
        settings["num_threads"] = num_threads
        options = cpu_session_options(**settings)
        self.session = ort.InferenceSession(str(path.parent / cfg["model_path"]), options,
                                            providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        expected = [1, cfg["input_channels"], cfg["input_height"], cfg["input_width"]]
        if len(inputs) != 1 or len(outputs) != 1 or inputs[0].name != cfg["input_name"] or outputs[0].name != cfg["output_name"]:
            raise ValueError("ONNX와 JSON 텐서 이름 불일치")
        if inputs[0].type != "tensor(float)" or outputs[0].type != "tensor(float)" or len(inputs[0].shape) != 4:
            raise ValueError("ONNX float32 NCHW 입력 필요")
        if inputs[0].shape[1] != cfg["input_channels"]:
            raise ValueError("ONNX와 JSON 입력 채널 불일치")
        if any(isinstance(v, int) and v != e for v, e in zip(inputs[0].shape, expected)):
            raise ValueError("ONNX와 JSON 입력 크기 불일치")
        if len(outputs[0].shape) != 2 or (isinstance(outputs[0].shape[1], int) and outputs[0].shape[1] != cfg["num_classes"]):
            raise ValueError("ONNX 클래스 개수 불일치")

    def logits(self, tensor):
        cfg = self.config
        result = self.session.run([cfg["output_name"]], {cfg["input_name"]: tensor})[0]
        if result.shape != (tensor.shape[0], cfg["num_classes"]) or not np.isfinite(result).all():
            raise ValueError("ONNX 출력 형태 또는 유한성 오류")
        return result
