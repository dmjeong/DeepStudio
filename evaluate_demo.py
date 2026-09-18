"""
Deep Vision Studio — 4 태스크 통합 데모 평가 스크립트

각 태스크별 유명 데이터셋 스타일의 합성 테스트 이미지 5장 생성 →
CustomCSP 모델 추론 → Grad-CAM 히트맵 → 결과 저장

├── Classification  (ImageNet 스타일)
├── Segmentation    (Pascal VOC 스타일)
├── Detection       (COCO 스타일)
└── Anomaly         (MVTec AD 스타일)

결과 구조:
    demo_results/
    ├── classification/
    │   ├── img_01_original.png
    │   ├── img_01_gradcam.png
    │   └── ...
    ├── segmentation/
    │   ├── img_01_original.png
    │   ├── img_01_prediction.png
    │   ├── img_01_gradcam.png
    │   └── ...
    ├── detection/
    │   └── ...
    ├── anomaly/
    │   └── ...
    └── summary_grid.png  ← 전체 요약 이미지
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# 프로젝트 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "gui"))

from model import CustomCSP
from core.gradcam import GradCAM


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "demo_results")
DEVICE = "cpu"
NUM_IMAGES = 5

# ── 태스크별 설정 ──
TASK_CONFIGS = {
    "classify": {
        "dataset_name": "ImageNet-1K",
        "num_classes": 10,
        "input_size": 224,
        "class_names": [
            "고양이", "강아지", "자동차", "비행기", "꽃",
            "새", "물고기", "나비", "배", "자전거"
        ],
    },
    "segment": {
        "dataset_name": "Pascal VOC 2012",
        "num_classes": 5,   # 배경 포함
        "input_size": 320,
        "class_names": ["배경", "사람", "자동차", "고양이", "나무"],
    },
    "detect": {
        "dataset_name": "MS COCO",
        "num_classes": 10,
        "input_size": 416,
        "class_names": [
            "사람", "자동차", "고양이", "강아지", "의자",
            "병", "책", "모니터", "컵", "가방"
        ],
    },
    "anomaly": {
        "dataset_name": "MVTec AD",
        "num_classes": 3,
        "input_size": 224,
        "class_names": ["채널1", "채널2", "채널3"],
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  합성 테스트 이미지 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_classification_images(size: int = 224, n: int = 5) -> list:
    """
    ImageNet 스타일 합성 이미지 생성

    각 이미지에 고유한 색상 패턴 + 기하학적 오브젝트 배치
    → 실제 ImageNet처럼 중앙에 주요 객체가 위치하는 구도
    """
    images = []
    # 각 이미지별 고유 컨셉
    concepts = [
        {"bg": (30, 80, 30), "shapes": "circles", "label": "cat-like"},
        {"bg": (80, 60, 30), "shapes": "rectangles", "label": "dog-like"},
        {"bg": (50, 50, 80), "shapes": "mixed", "label": "car-like"},
        {"bg": (70, 70, 90), "shapes": "triangles", "label": "plane-like"},
        {"bg": (30, 70, 50), "shapes": "petals", "label": "flower-like"},
    ]

    for i, c in enumerate(concepts[:n]):
        img = Image.new("RGB", (size, size), c["bg"])
        draw = ImageDraw.Draw(img)
        np.random.seed(42 + i)  # 재현 가능

        # 중앙 영역에 주요 오브젝트 패턴
        cx, cy = size // 2, size // 2

        if c["shapes"] == "circles":
            # 고양이 귀 + 얼굴 형태 시뮬레이션
            r = size // 4
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(180, 160, 120))
            draw.ellipse([cx-r, cy-r-r//2, cx-r//2, cy-r], fill=(200, 180, 140))  # 귀
            draw.ellipse([cx+r//2, cy-r-r//2, cx+r, cy-r], fill=(200, 180, 140))  # 귀
            draw.ellipse([cx-r//4, cy-r//6, cx-r//8, cy], fill=(60, 60, 40))  # 눈
            draw.ellipse([cx+r//8, cy-r//6, cx+r//4, cy], fill=(60, 60, 40))  # 눈

        elif c["shapes"] == "rectangles":
            # 강아지 형태 시뮬레이션
            r = size // 3
            draw.rounded_rectangle([cx-r, cy-r//2, cx+r, cy+r], radius=r//4, fill=(160, 130, 90))
            draw.ellipse([cx-r//3, cy-r//4, cx-r//6, cy], fill=(50, 40, 30))  # 눈
            draw.ellipse([cx+r//6, cy-r//4, cx+r//3, cy], fill=(50, 40, 30))  # 눈
            draw.ellipse([cx-r//6, cy+r//8, cx+r//6, cy+r//3], fill=(40, 30, 25))  # 코

        elif c["shapes"] == "mixed":
            # 자동차 형태 시뮬레이션
            draw.rectangle([cx-size//3, cy, cx+size//3, cy+size//5], fill=(180, 40, 40))
            draw.rectangle([cx-size//5, cy-size//6, cx+size//5, cy], fill=(180, 40, 40))
            draw.ellipse([cx-size//4, cy+size//8, cx-size//7, cy+size//4], fill=(40, 40, 40))  # 바퀴
            draw.ellipse([cx+size//7, cy+size//8, cx+size//4, cy+size//4], fill=(40, 40, 40))  # 바퀴

        elif c["shapes"] == "triangles":
            # 비행기 형태 시뮬레이션
            draw.polygon([(cx, cy-size//3), (cx-size//6, cy+size//6), (cx+size//6, cy+size//6)],
                        fill=(200, 200, 220))
            draw.polygon([(cx-size//3, cy+size//8), (cx, cy), (cx, cy+size//8)],
                        fill=(180, 180, 200))  # 왼쪽 날개
            draw.polygon([(cx+size//3, cy+size//8), (cx, cy), (cx, cy+size//8)],
                        fill=(180, 180, 200))  # 오른쪽 날개

        elif c["shapes"] == "petals":
            # 꽃 형태 시뮬레이션
            r = size // 6
            colors = [(255, 100, 100), (255, 150, 80), (255, 200, 100),
                     (200, 100, 200), (255, 120, 150)]
            for angle in range(0, 360, 72):
                rad = np.radians(angle)
                px = cx + int(r * 1.5 * np.cos(rad))
                py = cy + int(r * 1.5 * np.sin(rad))
                draw.ellipse([px-r, py-r, px+r, py+r], fill=colors[angle // 72])
            draw.ellipse([cx-r//2, cy-r//2, cx+r//2, cy+r//2], fill=(255, 255, 100))  # 중심

        # 약간의 노이즈 추가 (자연스러움)
        arr = np.array(img).astype(np.float32)
        noise = np.random.randn(*arr.shape) * 5
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        images.append(arr)

    return images


def create_segmentation_images(size: int = 320, n: int = 5) -> list:
    """
    Pascal VOC 스타일 합성 세그멘테이션 이미지 생성

    배경 + 전경 객체 (사람, 자동차 등) 구분 가능한 색상 영역
    """
    images = []
    scenes = [
        {"bg": (100, 180, 100), "obj_color": (200, 150, 120), "obj": "person"},
        {"bg": (150, 180, 220), "obj_color": (80, 80, 180), "obj": "car"},
        {"bg": (200, 200, 180), "obj_color": (180, 160, 120), "obj": "cat"},
        {"bg": (80, 120, 80), "obj_color": (60, 140, 60), "obj": "tree"},
        {"bg": (180, 180, 200), "obj_color": (180, 130, 100), "obj": "multi"},
    ]

    for i, scene in enumerate(scenes[:n]):
        img = Image.new("RGB", (size, size), scene["bg"])
        draw = ImageDraw.Draw(img)
        np.random.seed(100 + i)
        cx, cy = size // 2, size // 2

        # 하늘/땅 구분 (상단 1/3은 하늘)
        draw.rectangle([0, 0, size, size//3], fill=(135, 200, 235))

        if scene["obj"] == "person":
            # 사람 실루엣
            draw.ellipse([cx-25, cy-80, cx+25, cy-30], fill=scene["obj_color"])  # 머리
            draw.rectangle([cx-30, cy-30, cx+30, cy+60], fill=(50, 80, 150))  # 몸
            draw.rectangle([cx-15, cy+60, cx, cy+120], fill=(60, 60, 100))  # 왼다리
            draw.rectangle([cx, cy+60, cx+15, cy+120], fill=(60, 60, 100))  # 오른다리

        elif scene["obj"] == "car":
            draw.rectangle([cx-80, cy+10, cx+80, cy+60], fill=scene["obj_color"])
            draw.rectangle([cx-50, cy-20, cx+50, cy+10], fill=(100, 100, 200))
            draw.ellipse([cx-60, cy+40, cx-30, cy+70], fill=(40, 40, 40))
            draw.ellipse([cx+30, cy+40, cx+60, cy+70], fill=(40, 40, 40))

        elif scene["obj"] == "cat":
            r = size // 6
            draw.ellipse([cx-r, cy-r//2, cx+r, cy+r], fill=scene["obj_color"])
            draw.polygon([(cx-r, cy-r//2), (cx-r+15, cy-r), (cx-r//2, cy-r//2)], fill=scene["obj_color"])
            draw.polygon([(cx+r, cy-r//2), (cx+r-15, cy-r), (cx+r//2, cy-r//2)], fill=scene["obj_color"])

        elif scene["obj"] == "tree":
            draw.rectangle([cx-10, cy+20, cx+10, cy+80], fill=(100, 70, 40))
            for dy in range(-60, 20, 25):
                r = 40 - abs(dy) // 2
                draw.ellipse([cx-r, cy+dy-r, cx+r, cy+dy+r], fill=scene["obj_color"])

        elif scene["obj"] == "multi":
            # 사람 + 자동차
            draw.rectangle([cx+30, cy+10, cx+120, cy+50], fill=(80, 80, 180))
            draw.ellipse([cx+40, cy+35, cx+60, cy+55], fill=(40, 40, 40))
            draw.ellipse([cx+90, cy+35, cx+110, cy+55], fill=(40, 40, 40))
            draw.ellipse([cx-60, cy-60, cx-20, cy-20], fill=(200, 160, 130))
            draw.rectangle([cx-55, cy-20, cx-25, cy+40], fill=(50, 80, 150))

        arr = np.array(img).astype(np.float32)
        noise = np.random.randn(*arr.shape) * 3
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        images.append(arr)

    return images


def create_detection_images(size: int = 416, n: int = 5) -> list:
    """
    MS COCO 스타일 합성 디텍션 이미지 생성

    여러 객체가 배치된 실내/실외 장면
    """
    images = []
    scenes = [
        {"bg": (180, 180, 170), "name": "실내 장면"},
        {"bg": (100, 150, 100), "name": "공원"},
        {"bg": (120, 120, 140), "name": "도로"},
        {"bg": (200, 190, 170), "name": "사무실"},
        {"bg": (80, 130, 180), "name": "해변"},
    ]

    for i, scene in enumerate(scenes[:n]):
        img = Image.new("RGB", (size, size), scene["bg"])
        draw = ImageDraw.Draw(img)
        np.random.seed(200 + i)

        # 2~4개 객체 무작위 배치
        n_objects = np.random.randint(2, 5)
        obj_colors = [
            (200, 150, 120), (80, 80, 180), (180, 80, 80),
            (80, 180, 80), (180, 180, 80), (180, 80, 180)
        ]

        for j in range(n_objects):
            # 무작위 위치 & 크기
            w = np.random.randint(40, 100)
            h = np.random.randint(40, 120)
            x = np.random.randint(20, size - w - 20)
            y = np.random.randint(20, size - h - 20)
            color = obj_colors[j % len(obj_colors)]

            # 사각형 객체 + 약간의 디테일
            draw.rectangle([x, y, x+w, y+h], fill=color, outline=(0, 0, 0), width=2)

            # 내부 디테일 (원, 선 등)
            draw.ellipse([x+w//4, y+h//4, x+3*w//4, y+3*h//4],
                        fill=tuple(max(0, c-30) for c in color))

        arr = np.array(img).astype(np.float32)
        noise = np.random.randn(*arr.shape) * 4
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        images.append(arr)

    return images


def create_anomaly_images(size: int = 224, n: int = 5) -> list:
    """
    MVTec AD 스타일 합성 이상 탐지 이미지 생성

    정상 텍스처 + 일부에 결함(스크래치, 구멍 등) 추가
    """
    images = []
    textures = [
        {"base": (180, 180, 180), "pattern": "grid", "defect": False, "name": "정상-금속"},
        {"base": (200, 190, 170), "pattern": "wood", "defect": False, "name": "정상-나무"},
        {"base": (160, 160, 180), "pattern": "grid", "defect": True, "name": "결함-스크래치"},
        {"base": (190, 180, 160), "pattern": "wood", "defect": True, "name": "결함-구멍"},
        {"base": (170, 170, 170), "pattern": "uniform", "defect": True, "name": "결함-얼룩"},
    ]

    for i, tex in enumerate(textures[:n]):
        img = Image.new("RGB", (size, size), tex["base"])
        draw = ImageDraw.Draw(img)
        np.random.seed(300 + i)

        # 배경 텍스처 패턴
        if tex["pattern"] == "grid":
            for x in range(0, size, 20):
                draw.line([(x, 0), (x, size)], fill=(160, 160, 160), width=1)
            for y in range(0, size, 20):
                draw.line([(0, y), (size, y)], fill=(160, 160, 160), width=1)

        elif tex["pattern"] == "wood":
            for y in range(0, size, 8):
                offset = int(10 * np.sin(y / 30.0))
                color_var = int(15 * np.sin(y / 15.0))
                c = tuple(max(0, min(255, tex["base"][j] + color_var)) for j in range(3))
                draw.line([(offset, y), (size + offset, y)], fill=c, width=2)

        # 결함 추가
        if tex["defect"]:
            cx, cy = size // 2, size // 2

            if "스크래치" in tex["name"]:
                # 대각선 스크래치
                for offset in range(-5, 6, 2):
                    draw.line([(cx-50+offset, cy-40), (cx+60+offset, cy+50)],
                             fill=(80, 80, 80), width=2)

            elif "구멍" in tex["name"]:
                # 원형 결함
                draw.ellipse([cx-20, cy-15, cx+15, cy+20], fill=(50, 40, 30))
                draw.ellipse([cx-15, cy-10, cx+10, cy+15], fill=(30, 25, 20))

            elif "얼룩" in tex["name"]:
                # 불규칙 얼룩
                for _ in range(30):
                    dx = np.random.randint(-30, 30)
                    dy = np.random.randint(-30, 30)
                    r = np.random.randint(3, 8)
                    draw.ellipse([cx+dx-r, cy+dy-r, cx+dx+r, cy+dy+r],
                                fill=(100+np.random.randint(-20,20),
                                      80+np.random.randint(-20,20),
                                      60+np.random.randint(-20,20)))

        arr = np.array(img).astype(np.float32)
        noise = np.random.randn(*arr.shape) * 2
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        images.append(arr)

    return images


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  추론 + Grad-CAM 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def preprocess_image(image: np.ndarray, size: int) -> torch.Tensor:
    """
    이미지 전처리: numpy RGB → PyTorch 텐서 (정규화)

    변환 흐름:
    (H,W,3) uint8 → resize → (3,H,W) float32 → [0,1] → 정규화
    """
    from PIL import Image as PILImage

    # PIL로 리사이즈
    pil_img = PILImage.fromarray(image)
    pil_img = pil_img.resize((size, size), PILImage.BILINEAR)

    # numpy → tensor
    arr = np.array(pil_img).astype(np.float32) / 255.0  # [0, 1]

    # ImageNet 정규화
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std

    # (H, W, 3) → (1, 3, H, W)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def run_classification(images, config):
    """Classification 추론 + Grad-CAM"""
    print("\n" + "=" * 60)
    print(f"🔍 Classification — {config['dataset_name']}")
    print("=" * 60)

    model = CustomCSP(
        task="classify",
        num_classes=config["num_classes"],
        in_channels=3
    )
    model.eval()

    gradcam = GradCAM(model)
    results = []

    for i, img in enumerate(images):
        input_tensor = preprocess_image(img, config["input_size"])

        # 추론
        with torch.no_grad():
            output = model(input_tensor)

        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()
        pred_prob = probs[0, pred_class].item()

        # 상위 3개 예측
        top3_probs, top3_idx = probs[0].topk(3)

        # Grad-CAM 생성
        heatmap, overlay = gradcam.generate(input_tensor, img, alpha=0.5)

        class_name = config["class_names"][pred_class] if pred_class < len(config["class_names"]) else f"class_{pred_class}"

        result = {
            "original": img,
            "heatmap": heatmap,
            "overlay": overlay,
            "pred_class": pred_class,
            "pred_name": class_name,
            "pred_prob": pred_prob,
            "top3": [(config["class_names"][idx.item()] if idx.item() < len(config["class_names"]) else f"cls_{idx.item()}",
                      prob.item()) for idx, prob in zip(top3_idx, top3_probs)],
        }
        results.append(result)

        print(f"  이미지 {i+1}: 예측={class_name} (확률={pred_prob:.3f})")
        for name, prob in result["top3"]:
            print(f"    ├── {name}: {prob:.3f}")

    gradcam.release()
    return results


def run_segmentation(images, config):
    """Segmentation 추론 + Grad-CAM"""
    print("\n" + "=" * 60)
    print(f"🎨 Segmentation — {config['dataset_name']}")
    print("=" * 60)

    model = CustomCSP(
        task="segment",
        num_classes=config["num_classes"],
        in_channels=3
    )
    model.eval()

    gradcam = GradCAM(model)
    results = []

    # 세그멘테이션 컬러 팔레트 (Pascal VOC 스타일)
    palette = np.array([
        [0, 0, 0],        # 배경 (검정)
        [128, 0, 0],      # 클래스1 (빨강)
        [0, 128, 0],      # 클래스2 (초록)
        [128, 128, 0],    # 클래스3 (노랑)
        [0, 0, 128],      # 클래스4 (파랑)
    ], dtype=np.uint8)

    for i, img in enumerate(images):
        input_tensor = preprocess_image(img, config["input_size"])

        # 추론 — 출력: (B, C, H, W)
        with torch.no_grad():
            output = model(input_tensor)

        # 예측 마스크
        pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()  # (H, W)

        # 컬러 마스크 생성
        h, w = pred_mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id in range(config["num_classes"]):
            color_mask[pred_mask == cls_id] = palette[cls_id % len(palette)]

        # 클래스별 비율 계산
        unique, counts = np.unique(pred_mask, return_counts=True)
        total_pixels = pred_mask.size
        class_ratios = {}
        for cls_id, count in zip(unique, counts):
            name = config["class_names"][cls_id] if cls_id < len(config["class_names"]) else f"cls_{cls_id}"
            class_ratios[name] = count / total_pixels

        # Grad-CAM 생성
        heatmap, overlay = gradcam.generate(input_tensor, img, alpha=0.5)

        result = {
            "original": img,
            "prediction": color_mask,
            "heatmap": heatmap,
            "overlay": overlay,
            "class_ratios": class_ratios,
        }
        results.append(result)

        print(f"  이미지 {i+1}: 클래스 분포")
        for name, ratio in sorted(class_ratios.items(), key=lambda x: -x[1]):
            bar = "█" * int(ratio * 20)
            print(f"    ├── {name}: {ratio*100:.1f}% {bar}")

    gradcam.release()
    return results


def run_detection(images, config):
    """Detection 추론 + Grad-CAM"""
    print("\n" + "=" * 60)
    print(f"📦 Detection — {config['dataset_name']}")
    print("=" * 60)

    model = CustomCSP(
        task="detect",
        num_classes=config["num_classes"],
        in_channels=3
    )
    model.eval()

    gradcam = GradCAM(model)
    results = []

    for i, img in enumerate(images):
        input_tensor = preprocess_image(img, config["input_size"])

        # 추론 — 출력: (B, N, 5+C)
        with torch.no_grad():
            output = model(input_tensor)

        # NMS 간이 구현 (상위 5개 박스)
        preds = output[0]  # (N, 5+C)
        obj_scores = torch.sigmoid(preds[:, 4])
        cls_scores = torch.sigmoid(preds[:, 5:])

        # confidence = objectness × max_class_prob
        max_cls_scores, max_cls_ids = cls_scores.max(dim=1)
        confidences = obj_scores * max_cls_scores

        # 상위 5개
        top_k = min(5, len(confidences))
        top_conf, top_idx = confidences.topk(top_k)

        # 박스 좌표 (cx, cy, w, h → x1, y1, x2, y2)
        boxes = []
        for j in range(top_k):
            idx = top_idx[j].item()
            cx, cy, w, h = preds[idx, :4].tolist()
            x1, y1 = cx - w/2, cy - h/2
            x2, y2 = cx + w/2, cy + h/2
            cls_id = max_cls_ids[idx].item()
            conf = top_conf[j].item()

            cls_name = config["class_names"][cls_id] if cls_id < len(config["class_names"]) else f"cls_{cls_id}"
            boxes.append({
                "bbox": [x1, y1, x2, y2],
                "class": cls_name,
                "class_id": cls_id,
                "confidence": conf,
            })

        # 박스 그린 이미지 생성
        det_img = img.copy()
        pil_det = Image.fromarray(det_img)
        draw = ImageDraw.Draw(pil_det)

        box_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255)
        ]

        h_img, w_img = img.shape[:2]
        for j, box in enumerate(boxes):
            # 좌표를 이미지 크기로 스케일링
            x1 = max(0, min(w_img-1, int(box["bbox"][0] * w_img / config["input_size"])))
            y1 = max(0, min(h_img-1, int(box["bbox"][1] * h_img / config["input_size"])))
            x2 = max(0, min(w_img-1, int(box["bbox"][2] * w_img / config["input_size"])))
            y2 = max(0, min(h_img-1, int(box["bbox"][3] * h_img / config["input_size"])))

            color = box_colors[j % len(box_colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, max(0, y1-12)), f"{box['class']} {box['confidence']:.2f}", fill=color)

        det_array = np.array(pil_det)

        # Grad-CAM 생성
        heatmap, overlay = gradcam.generate(input_tensor, img, alpha=0.5)

        result = {
            "original": img,
            "detection": det_array,
            "heatmap": heatmap,
            "overlay": overlay,
            "boxes": boxes,
        }
        results.append(result)

        print(f"  이미지 {i+1}: {len(boxes)}개 객체 탐지")
        for box in boxes:
            print(f"    ├── {box['class']}: conf={box['confidence']:.3f} "
                  f"bbox=[{box['bbox'][0]:.0f},{box['bbox'][1]:.0f},{box['bbox'][2]:.0f},{box['bbox'][3]:.0f}]")

    gradcam.release()
    return results


def run_anomaly(images, config):
    """Anomaly Detection 추론 + Grad-CAM"""
    print("\n" + "=" * 60)
    print(f"🔬 Anomaly Detection — {config['dataset_name']}")
    print("=" * 60)

    model = CustomCSP(
        task="anomaly",
        num_classes=config["num_classes"],
        in_channels=3
    )
    model.eval()

    gradcam = GradCAM(model)
    results = []

    for i, img in enumerate(images):
        input_tensor = preprocess_image(img, config["input_size"])

        # 추론 — 출력: (B, C, H, W) 재구성 이미지
        with torch.no_grad():
            output = model(input_tensor)

        # 재구성 오차 맵 (MSE per pixel)
        error_map = (input_tensor - output).pow(2).mean(dim=1)  # (B, H, W)
        error_map = error_map.squeeze().cpu().numpy()

        # 정규화 → 히트맵
        e_min, e_max = error_map.min(), error_map.max()
        if e_max - e_min > 1e-8:
            error_norm = (error_map - e_min) / (e_max - e_min)
        else:
            error_norm = np.zeros_like(error_map)

        # 이상 점수 (평균 재구성 오차)
        anomaly_score = float(error_map.mean())

        # 오차 맵 시각화 (JET 컬러맵)
        error_vis = GradCAM._jet_colormap(error_norm)
        # 원본 크기로 리사이즈
        error_resized = GradCAM._resize_cam(error_norm, (img.shape[1], img.shape[0]))
        error_vis = GradCAM._jet_colormap(error_resized)

        # Grad-CAM 생성
        heatmap, overlay = gradcam.generate(input_tensor, img, alpha=0.5)

        result = {
            "original": img,
            "error_map": error_vis,
            "heatmap": heatmap,
            "overlay": overlay,
            "anomaly_score": anomaly_score,
            "is_anomaly": anomaly_score > 0.5,  # 임계값 (미학습이므로 참고용)
        }
        results.append(result)

        status = "⚠️ 이상" if result["is_anomaly"] else "✅ 정상"
        print(f"  이미지 {i+1}: 이상 점수={anomaly_score:.4f} → {status}")

    gradcam.release()
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  결과 시각화 (Summary Grid)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_individual_results(all_results: dict, output_dir: str):
    """태스크별 개별 결과 이미지 저장"""
    for task, results in all_results.items():
        task_dir = os.path.join(output_dir, task)
        os.makedirs(task_dir, exist_ok=True)

        for i, res in enumerate(results):
            # 원본
            Image.fromarray(res["original"]).save(
                os.path.join(task_dir, f"img_{i+1:02d}_original.png")
            )
            # Grad-CAM 오버레이
            Image.fromarray(res["overlay"]).save(
                os.path.join(task_dir, f"img_{i+1:02d}_gradcam.png")
            )
            # 태스크별 추가 결과
            if task == "segment" and "prediction" in res:
                Image.fromarray(res["prediction"]).save(
                    os.path.join(task_dir, f"img_{i+1:02d}_segmask.png")
                )
            elif task == "detect" and "detection" in res:
                Image.fromarray(res["detection"]).save(
                    os.path.join(task_dir, f"img_{i+1:02d}_detection.png")
                )
            elif task == "anomaly" and "error_map" in res:
                Image.fromarray(res["error_map"]).save(
                    os.path.join(task_dir, f"img_{i+1:02d}_error_map.png")
                )


def create_summary_grid(all_results: dict, output_path: str):
    """
    4 태스크 × 5 이미지 요약 그리드 생성

    레이아웃:
    ┌──────────────────────────────────────────────────────┐
    │                    CustomCSP Demo Results               │
    ├──────────┬────────┬────────┬────────┬────────────────┤
    │  Task    │ Img 1  │ Img 2  │ Img 3  │ Img 4 │ Img 5 │
    ├──────────┼────────┼────────┼────────┼───────┼───────┤
    │ Classify │ Orig   │ Orig   │ ...    │       │       │
    │          │ GradCAM│ GradCAM│ ...    │       │       │
    ├──────────┼────────┼────────┼────────┼───────┼───────┤
    │ Segment  │ Orig   │ ...                             │
    │          │ Mask   │                                  │
    │          │ GradCAM│                                  │
    ├──────────┼────────┼────────┼────────┼───────┼───────┤
    │ Detect   │ ...                                      │
    ├──────────┼────────┼────────┼────────┼───────┼───────┤
    │ Anomaly  │ ...                                      │
    └──────────┴────────┴────────┴────────┴───────┴───────┘
    """
    # 셀 크기
    CELL_SIZE = 160       # 각 이미지 셀
    LABEL_W = 180         # 왼쪽 태스크 레이블 너비
    HEADER_H = 60         # 상단 헤더 높이
    PADDING = 4           # 셀 간격
    TEXT_H = 20           # 텍스트 영역

    # 태스크별 행 수 (원본 + 추가 시각화 + GradCAM)
    task_rows = {
        "classify": 2,    # 원본, GradCAM
        "segment": 3,     # 원본, Seg Mask, GradCAM
        "detect": 3,      # 원본, Detection, GradCAM
        "anomaly": 3,     # 원본, Error Map, GradCAM
    }

    # 전체 이미지 크기 계산
    total_cols = NUM_IMAGES
    total_w = LABEL_W + total_cols * (CELL_SIZE + PADDING) + PADDING
    total_h = HEADER_H
    for task in ["classify", "segment", "detect", "anomaly"]:
        total_h += task_rows[task] * (CELL_SIZE + TEXT_H + PADDING) + PADDING + 30  # 태스크 구분

    # 캔버스 생성
    canvas = Image.new("RGB", (total_w, total_h), (32, 32, 40))
    draw = ImageDraw.Draw(canvas)

    # 헤더
    draw.rectangle([0, 0, total_w, HEADER_H], fill=(45, 45, 55))
    draw.text((total_w // 2 - 200, 15), "Deep Vision Studio — 4-Task Demo Evaluation",
              fill=(220, 220, 240))
    draw.text((total_w // 2 - 150, 35), "(Untrained Model · Pipeline Verification)",
              fill=(150, 150, 170))

    # 각 태스크 렌더링
    y_offset = HEADER_H + PADDING

    task_info = {
        "classify": {"icon": "🔍", "name": "Classification", "dataset": "ImageNet-1K",
                     "rows": ["Original", "Grad-CAM"]},
        "segment":  {"icon": "🎨", "name": "Segmentation", "dataset": "Pascal VOC",
                     "rows": ["Original", "Seg Mask", "Grad-CAM"]},
        "detect":   {"icon": "📦", "name": "Detection", "dataset": "MS COCO",
                     "rows": ["Original", "Detection", "Grad-CAM"]},
        "anomaly":  {"icon": "🔬", "name": "Anomaly", "dataset": "MVTec AD",
                     "rows": ["Original", "Error Map", "Grad-CAM"]},
    }

    for task_key in ["classify", "segment", "detect", "anomaly"]:
        info = task_info[task_key]
        results = all_results[task_key]

        # 태스크 제목 바
        draw.rectangle([0, y_offset, total_w, y_offset + 25], fill=(55, 55, 70))
        draw.text((10, y_offset + 4),
                  f"{info['icon']} {info['name']}  ({info['dataset']})",
                  fill=(255, 200, 100))
        y_offset += 30

        # 행 이름 목록
        row_names = info["rows"]

        for row_idx, row_name in enumerate(row_names):
            # 행 레이블
            ry = y_offset + row_idx * (CELL_SIZE + TEXT_H + PADDING)
            draw.text((10, ry + CELL_SIZE // 2), row_name, fill=(180, 180, 200))

            # 각 이미지 셀
            for col_idx, res in enumerate(results):
                cx = LABEL_W + col_idx * (CELL_SIZE + PADDING) + PADDING
                cy = ry

                # 이미지 선택
                if row_name == "Original":
                    cell_img = res["original"]
                elif row_name == "Grad-CAM":
                    cell_img = res["overlay"]
                elif row_name == "Seg Mask":
                    cell_img = res.get("prediction", res["original"])
                elif row_name == "Detection":
                    cell_img = res.get("detection", res["original"])
                elif row_name == "Error Map":
                    cell_img = res.get("error_map", res["original"])
                else:
                    cell_img = res["original"]

                # 리사이즈 & 붙이기
                pil_cell = Image.fromarray(cell_img).resize(
                    (CELL_SIZE, CELL_SIZE), Image.BILINEAR
                )
                canvas.paste(pil_cell, (cx, cy))

                # 셀 아래 주석
                if row_name == "Original" and task_key == "classify":
                    text = f"{res['pred_name']} ({res['pred_prob']:.2f})"
                    draw.text((cx, cy + CELL_SIZE + 2), text, fill=(200, 200, 220))
                elif row_name == "Original" and task_key == "detect":
                    text = f"{len(res['boxes'])} objs"
                    draw.text((cx, cy + CELL_SIZE + 2), text, fill=(200, 200, 220))
                elif row_name == "Original" and task_key == "anomaly":
                    score = res['anomaly_score']
                    status = "Anomaly" if res['is_anomaly'] else "Normal"
                    color = (255, 100, 100) if res['is_anomaly'] else (100, 255, 100)
                    draw.text((cx, cy + CELL_SIZE + 2), f"{status} ({score:.3f})", fill=color)

        y_offset += len(row_names) * (CELL_SIZE + TEXT_H + PADDING) + PADDING

    # 저장
    canvas.save(output_path, quality=95)
    print(f"\n📊 요약 그리드 저장: {output_path}")
    print(f"   크기: {canvas.size[0]}×{canvas.size[1]} px")
    return output_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  메인 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Deep Vision Studio — 4-Task Demo Evaluation      ║")
    print("║  각 태스크 × 5개 이미지 → 추론 + Grad-CAM 시각화     ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print("⚠️  미학습 모델 (Kaiming 초기화) → 예측은 무작위입니다.")
    print("    이 데모는 파이프라인이 정상 작동하는지 검증합니다.")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. 합성 테스트 이미지 생성 ──
    print("📷 합성 테스트 이미지 생성 중...")
    cls_images = create_classification_images(224, NUM_IMAGES)
    seg_images = create_segmentation_images(320, NUM_IMAGES)
    det_images = create_detection_images(416, NUM_IMAGES)
    ano_images = create_anomaly_images(224, NUM_IMAGES)
    print(f"   ✅ 총 {NUM_IMAGES * 4}개 이미지 생성 완료")

    # ── 2. 태스크별 추론 + Grad-CAM ──
    all_results = {}

    all_results["classify"] = run_classification(cls_images, TASK_CONFIGS["classify"])
    all_results["segment"]  = run_segmentation(seg_images, TASK_CONFIGS["segment"])
    all_results["detect"]   = run_detection(det_images, TASK_CONFIGS["detect"])
    all_results["anomaly"]  = run_anomaly(ano_images, TASK_CONFIGS["anomaly"])

    # ── 3. 결과 저장 ──
    print("\n" + "=" * 60)
    print("💾 결과 저장 중...")
    save_individual_results(all_results, OUTPUT_DIR)

    grid_path = os.path.join(OUTPUT_DIR, "summary_grid.png")
    create_summary_grid(all_results, grid_path)

    # ── 4. 요약 통계 ──
    print("\n" + "=" * 60)
    print("📊 태스크별 요약")
    print("=" * 60)

    for task, results in all_results.items():
        config = TASK_CONFIGS[task]
        model = CustomCSP(
            task=task,
            num_classes=config["num_classes"],
            in_channels=3
        )
        params = model.get_param_count()
        print(f"\n  {task.upper()}")
        print(f"    ├── 데이터셋: {config['dataset_name']}")
        print(f"    ├── 모델 크기: {params['total']:,} params ({params['total_MB']:.1f} MB)")
        print(f"    ├── 입력 크기: {config['input_size']}×{config['input_size']}")
        print(f"    ├── 클래스 수: {config['num_classes']}")
        print(f"    └── 이미지 수: {len(results)}개 처리 완료 ✅")

    print(f"\n🎉 전체 평가 완료! 결과: {OUTPUT_DIR}/")
    print(f"   요약 그리드: {grid_path}")


if __name__ == "__main__":
    main()
