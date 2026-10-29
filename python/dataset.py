"""
CustomCSP 데이터셋 로더
- Classification: 폴더 기반 이미지 분류 (ImageFolder 형식)
- Segmentation: 이미지 + 마스크 쌍 로딩
- 공통 전처리 & 데이터 증강 파이프라인

디렉토리 구조:
┌───────────────────────────────────────────────────────────┐
│  Classification 데이터 구조        Segmentation 데이터 구조  │
├───────────────────────────────────────────────────────────┤
│  data/                            data/                    │
│  ├── train/                       ├── images/              │
│  │   ├── class_A/                 │   ├── train/           │
│  │   │   ├── img001.jpg           │   │   ├── img001.jpg   │
│  │   │   └── img002.jpg           │   │   └── img002.jpg   │
│  │   ├── class_B/                 │   └── val/             │
│  │   │   └── ...                  │       └── ...          │
│  │   └── class_C/                 ├── masks/               │
│  │       └── ...                  │   ├── train/           │
│  └── val/                         │   │   ├── img001.png   │
│      ├── class_A/                 │   │   └── img002.png   │
│      └── ...                      │   └── val/             │
│                                   │       └── ...          │
└───────────────────────────────────────────────────────────┘
"""

import os
import random
from collections import OrderedDict
from typing import Tuple, List, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
from PIL import Image
from opencv_preprocess import OpenCVResize, read_image, resize
from checkpoint import PATCH_SCALE_INTERPOLATION


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  데이터 증강 파이프라인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── 채널별 정규화 상수 ──────────────────────────────
# 3ch (RGB): ImageNet 표준 정규화 값
IMAGENET_MEAN_3CH = [0.485, 0.456, 0.406]
IMAGENET_STD_3CH = [0.229, 0.224, 0.225]

# 1ch: 프로젝트에서 사용해 온 정규화 상수다.
# 실제 데이터의 통계 추정값이나 공식 ImageNet grayscale 통계를 뜻하지 않는다.
IMAGENET_MEAN_1CH = [0.449]
IMAGENET_STD_1CH = [0.226]


def get_normalize_params(in_channels: int = 3):
    """
    채널 수에 맞는 정규화 mean/std 반환

    Args:
        in_channels: 1 (Grayscale) 또는 3 (RGB)

    Returns:
        (mean, std) — 각각 리스트
    """
    if in_channels == 1:
        return IMAGENET_MEAN_1CH, IMAGENET_STD_1CH
    return IMAGENET_MEAN_3CH, IMAGENET_STD_3CH


def get_classification_transforms(
    input_size: Tuple[int, int] = (224, 224),
    is_train: bool = True,
    in_channels: int = 3,
    flip_prob: float = 0.5,
    rotation: float = 15.0,
    color_jitter: float = 0.2
) -> T.Compose:
    """
    Classification 전용 데이터 증강 파이프라인

    학습 시:
        OpenCV Resize → ToTensor → RandomHFlip → RandomRotation → [ColorJitter] → Normalize
    검증 시:
        Resize → ToTensor → Normalize

    Note: 1ch (Grayscale) 모드에서는 ColorJitter의 saturation/hue 제거
          (그레이스케일에는 색상 정보가 없음)
    """
    # 채널 수에 맞는 정규화 값 선택
    mean, std = get_normalize_params(in_channels)

    if is_train:
        augmentations = [
            OpenCVResize(input_size),
            T.ToTensor(),
            T.RandomHorizontalFlip(p=flip_prob),
            T.RandomRotation(degrees=rotation),
        ]

        # 1ch: brightness/contrast만 적용 (saturation/hue는 무의미)
        if in_channels == 1:
            augmentations.append(T.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
            ))
        else:
            augmentations.append(T.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=color_jitter / 2  # hue는 범위가 작음
            ))

        augmentations += [
            T.Normalize(mean=mean, std=std),
        ]
        return T.Compose(augmentations)
    else:
        return T.Compose([
            OpenCVResize(input_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Transform 교체 래퍼 (random_split val 전용)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _TransformOverrideSubset(Dataset):
    """
    random_split이 반환하는 Subset의 transform을 교체하는 래퍼

    ┌──────────────────────────────────────────────────────┐
    │ 문제: random_split → Subset은 원본 Dataset의 transform│
    │       을 그대로 사용 → val에 학습 증강이 적용됨        │
    │                                                      │
    │ 해결: 원본 Dataset에서 이미지+레이블을 가져온 뒤        │
    │       새 transform을 적용                             │
    └──────────────────────────────────────────────────────┘

    Args:
        subset: torch.utils.data.Subset (random_split 결과)
        transform: 교체할 새 transform (증강 없는 검증용)
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        # 상위 코드에서 .dataset.classes 등에 접근할 수 있도록 위임
        self.dataset = subset.dataset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        """원본 Dataset의 raw 이미지를 새 transform으로 처리"""
        # Subset.indices[idx]로 원본 인덱스 획득
        original_idx = self.subset.indices[idx]
        base_dataset = self.subset.dataset

        # 원본 픽셀을 OpenCV로 읽고 검증용 변환 적용
        img_path, label = base_dataset.samples[original_idx]
        image = base_dataset.read_cached(img_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    @property
    def classes(self):
        return self.subset.dataset.classes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Classification 데이터셋
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ClassificationDataset(Dataset):
    """
    폴더 기반 이미지 분류 데이터셋

    폴더명이 클래스명이 됨 (ImageFolder 형식)

    Args:
        root: 데이터 루트 경로 (하위에 클래스별 폴더)
        transform: 이미지 변환 파이프라인
        in_channels: 입력 채널 수 (1=Grayscale, 3=RGB)
        extensions: 지원 이미지 확장자
    """

    # 지원 이미지 확장자
    SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    def __init__(self, root: str, transform=None,
                 in_channels: int = 3,
                 extensions: set = None, class_to_idx: Optional[Dict[str, int]] = None,
                 cache_images: bool = True, cache_size_mb: int = 32):
        super().__init__()
        self.root = root
        self.transform = transform
        self.in_channels = in_channels
        self.extensions = extensions or self.SUPPORTED_EXT
        # 각 DataLoader worker가 디코딩 결과를 제한된 크기로 재사용한다.
        # 원본 해상도가 큰 데이터에서도 메모리가 무한히 증가하지 않는다.
        self.cache_images = bool(cache_images)
        self.cache_limit_bytes = max(0, int(cache_size_mb)) * 1024 * 1024
        self._image_cache = {}
        self._image_cache_bytes = 0

        # 클래스 목록 & 인덱스 매핑 구축
        self.classes, self.class_to_idx = self._find_classes()
        if class_to_idx is not None:
            unknown = set(self.classes) - set(class_to_idx)
            if unknown:
                raise ValueError(f"학습에 없는 검증 클래스: {sorted(unknown)}")
            self.class_to_idx = dict(class_to_idx)
            self.classes = sorted(class_to_idx, key=class_to_idx.get)
        # 이미지 경로 & 레이블 목록
        self.samples = self._make_dataset()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"데이터가 없습니다: {root}\n"
                f"폴더 구조: root/class_name/image.jpg 형식이어야 합니다."
            )

        print(f"Classification 데이터셋 로드: {root}")
        print(f"   클래스 수: {len(self.classes)}")
        print(f"   이미지 수: {len(self.samples)}")
        for cls_name, idx in self.class_to_idx.items():
            count = sum(1 for _, label in self.samples if label == idx)
            print(f"   - {cls_name}: {count}장")

    def _find_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """하위 폴더명을 클래스로 인식"""
        classes = sorted([
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        ])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self) -> List[Tuple[str, int]]:
        """이미지 경로와 레이블 쌍 목록 생성"""
        samples = []
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = os.path.join(self.root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.extensions:
                    path = os.path.join(cls_dir, fname)
                    samples.append((path, cls_idx))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def read_cached(self, path: str) -> np.ndarray:
        """Read once per worker and retain decoded pixels in a bounded LRU cache."""
        key = os.fspath(path)
        cached = self._image_cache.pop(key, None)
        if cached is not None:
            self._image_cache[key] = cached
            return cached
        image = read_image(key, self.in_channels)
        size = int(image.nbytes)
        if self.cache_images and self.cache_limit_bytes and size <= self.cache_limit_bytes:
            while self._image_cache and self._image_cache_bytes + size > self.cache_limit_bytes:
                oldest = next(iter(self._image_cache))
                removed = self._image_cache.pop(oldest)
                self._image_cache_bytes -= int(removed.nbytes)
            self._image_cache[key] = image
            self._image_cache_bytes += size
        return image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        # 이미지 로드 — 채널 수에 따라 RGB 또는 Grayscale 변환
        image = self.read_cached(path)
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_names(self) -> List[str]:
        """클래스 이름 목록 반환"""
        return self.classes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Segmentation 데이터셋
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def resize_label_mask(mask: np.ndarray, size: Tuple[int, int], *, keep_ratio: float = 0.5) -> np.ndarray:
    """클래스 번호 마스크를 (H, W)로 바꾼다. 축소해도 작은 결함이 정답에서 사라지지 않게 한다.

    1. 픽셀 중심 nearest(INTER_NEAREST_EXACT): 이미지의 INTER_LINEAR_EXACT·INTER_AREA(패치 배율)와 같은 격자.
       INTER_NEAREST(왼쪽 위 기준)는 4배 축소에서 정답을 평균 0.38 출력 px 밀어낸다.
    2. nearest는 블록마다 한 픽셀만 읽어서, 블록보다 얇은 결함은 통째로 사라지거나
       (4배 축소의 3×3 점 결함 약 43%) 점선으로 끊긴다(비스듬한 1px 흠집 94%, 2px 70%).
       클래스별로 출력 픽셀이 덮는 원본 면적 비율(coverage)을 구하고, coverage > 0인
       출력 영역의 연결 성분마다 다음이면 그 성분의 배경 출력 픽셀을 해당 클래스로 되살린다.
         - nearest가 남긴 픽셀 수 < 면적상 기대치(coverage 합) × keep_ratio  (사라짐)
         - 블록보다 얇은 성분(평균 coverage < 0.5)인데 nearest 결과가 2조각 이상  (끊김)
       되살린 얇은 선은 출력 1~2px 폭의 연결된 선이 된다(출력 격자에서 가능한 최소 폭).
       충분히 남은 성분(큰 결함은 평균 coverage 0.8 이상)과 다른 클래스·255(학습 제외)
       픽셀은 바꾸지 않는다.

    연결 성분은 출력 해상도에서 구한다: 원본에서 붙어 있는 결함은 출력에서도 붙어 있고,
    원본 해상도 연결 성분(4096²에서 클래스당 약 50ms)을 피한다.
    """
    import cv2

    height, width = int(size[0]), int(size[1])
    resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST_EXACT)
    if mask.shape[0] <= height and mask.shape[1] <= width:  # 확대는 성분을 잃지 않는다
        return resized
    if mask.dtype == np.uint8:  # 정렬하는 np.unique(4096²에서 약 0.1초) 대신 히스토그램
        histogram = cv2.calcHist([mask], [0], None, [256], [0, 256]).ravel()
        classes = np.flatnonzero(histogram)
    else:
        classes = np.unique(mask)
    # 출력 픽셀 하나가 덮는 원본 픽셀 수. 255 이하이면 uint8(0/255) 면적 평균에서도
    # 원본 1픽셀이 반올림 후 1 이상으로 남는다. 더 큰 블록은 uint16(0/65535)으로 정밀도를 높인다.
    block = (mask.shape[0] / height) * (mask.shape[1] / width)
    full = 255 if block <= 255 else 65535
    for cls in classes:
        if cls == 0 or cls == 255:
            continue
        # 출력 픽셀별 해당 클래스 면적 비율 × full (0이면 블록 안에 그 클래스 픽셀이 없다)
        member = cv2.inRange(mask, int(cls), int(cls))  # uint8 0/255, 원본 해상도에서 가장 싼 비교
        if full != 255:
            member = member.astype(np.uint16) * np.uint16(257)
        area = cv2.resize(member, (width, height), interpolation=cv2.INTER_AREA)
        coverage = area.astype(np.float32) * (1.0 / full)  # 출력 해상도라 변환 비용이 작다
        present = (area > 0).astype(np.uint8)
        count, labels = cv2.connectedComponents(present, connectivity=8)
        if count <= 1:
            continue
        flat = labels.ravel()
        expected = np.bincount(flat, weights=coverage.ravel(), minlength=count)
        pixels = np.bincount(flat, minlength=count)
        sampled = (resized == cls).astype(np.uint8)
        kept = np.bincount(flat, weights=sampled.ravel(), minlength=count)
        # nearest가 남긴 조각 수: 조각마다 그것이 속한 coverage 성분 번호를 찾아 센다
        piece_count, piece_labels = cv2.connectedComponents(sampled, connectivity=8)
        owner = np.zeros(piece_count, dtype=np.int64)
        owner[piece_labels.ravel()] = flat  # 한 조각의 픽셀은 모두 같은 성분에 속한다
        pieces = np.bincount(owner[1:], minlength=count)
        thin = expected < 0.5 * pixels
        lost = np.flatnonzero((kept < keep_ratio * expected) | (thin & (pieces > 1)))
        lost = lost[lost != 0]  # 0은 coverage가 없는 영역
        if not lost.size:
            continue
        restore = np.isin(labels, lost) & (resized == 0)
        resized[restore] = cls
    return resized


class SegmentationDataset(Dataset):
    """
    이미지 + 마스크 세그멘테이션 데이터셋

    마스크: 단일 채널, 픽셀값 = 클래스 인덱스 (0: 배경)

    Args:
        img_dir: 이미지 디렉토리
        mask_dir: 마스크 디렉토리
        input_size: 리사이즈 크기 (H, W)
        is_train: 학습 모드 여부
        flip_prob: 수평 뒤집기 확률
    """
    SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    def __init__(self, img_dir: str, mask_dir: str,
                 input_size: Tuple[int, int] = (320, 320),
                 in_channels: int = 3,
                 is_train: bool = True,
                 flip_prob: float = 0.5, rotation: float = 0.0,
                 color_jitter: float = 0.0, num_classes: Optional[int] = None):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.input_size = input_size
        self.in_channels = in_channels
        self.is_train = is_train
        self.flip_prob = flip_prob
        self.rotation = rotation
        self.color_jitter = color_jitter
        self.num_classes = num_classes

        # 채널 수에 맞는 정규화 값 선택
        self.mean, self.std = get_normalize_params(in_channels)

        # 축소 마스크 캐시: 작은 결함 보존 계산(2048² 약 25ms, 4096² 약 50ms)을 첫 epoch에만 한다.
        # 파일 수정 시각·크기가 키라 바뀐 마스크는 다시 계산한다. 작업자 프로세스마다 64MB 한도.
        self._mask_cache = OrderedDict()
        self._mask_cache_bytes = 0
        self._mask_cache_limit = 64 * 1024 * 1024

        # 이미지-마스크 쌍 매칭
        self.unlabeled = []
        self.pairs = self._find_pairs()

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"이미지-마스크 쌍을 찾을 수 없습니다.\n"
                f"  이미지: {img_dir}\n"
                f"  마스크: {mask_dir}"
            )

        print("Segmentation 데이터셋 로드")
        print(f"   이미지-마스크 쌍: {len(self.pairs)}")
        if self.unlabeled:
            print(f"   마스크 없는 이미지 {len(self.unlabeled)}장은 배경(결함 없음)으로 학습")

    def _find_pairs(self) -> List[Tuple[str, str]]:
        from spatial_data import labeled_semantic_pairs
        # 마스크가 없는 이미지(라벨링 안 함)는 결함이 없는 이미지로 보고 전부 배경(0)인 정답으로 학습한다.
        # 쌍의 마스크 자리는 None이다 (_resized_mask가 배경 마스크를 만든다).
        pairs, self.unlabeled = labeled_semantic_pairs(self.img_dir, self.mask_dir)
        if not pairs:
            raise ValueError(
                f"정답 마스크가 하나도 없습니다: {self.img_dir} (이미지 {len(self.unlabeled)}장). "
                "데이터셋 화면에서 이미지를 열어 결함 영역을 그리고 저장하세요.")
        return pairs + [(image, None) for image in self.unlabeled]

    def _resized_mask(self, mask_path: Optional[str]) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        """학습 크기로 줄인 클래스 마스크(복사본)와 원본 (W, H). 마스크 없는 이미지는 전부 배경 (크기 None)."""
        if mask_path is None:
            return np.zeros(tuple(self.input_size), dtype=np.uint8), None
        stat = os.stat(mask_path)
        key = (mask_path, stat.st_mtime_ns, stat.st_size)
        cached = self._mask_cache.get(key)
        if cached is None:
            # 팔레트 마스크는 표시 색이 아닌 인덱스를 보존한다.
            with Image.open(mask_path) as source_mask:
                if source_mask.mode not in ("P", "L", "I", "I;16"):
                    raise ValueError(
                        f"정답 마스크는 클래스 인덱스 이미지 필요 (팔레트/L/I): {mask_path}"
                    )
                original = np.asarray(source_mask)
                size = source_mask.size
            cached = (resize_label_mask(original, self.input_size), size)
            nbytes = cached[0].nbytes
            if nbytes <= self._mask_cache_limit:
                while self._mask_cache and self._mask_cache_bytes + nbytes > self._mask_cache_limit:
                    _, (dropped, _) = self._mask_cache.popitem(last=False)
                    self._mask_cache_bytes -= dropped.nbytes
                self._mask_cache[key] = cached
                self._mask_cache_bytes += nbytes
        else:
            self._mask_cache.move_to_end(key)
        return cached[0].copy(), cached[1]

    def class_pixel_counts(self, num_classes: int) -> List[int]:
        """학습 크기 마스크의 클래스별 픽셀 수 (255 학습 제외 픽셀은 세지 않음)."""
        counts = np.zeros(num_classes, dtype=np.int64)
        for _, mask_path in self.pairs:
            mask = self._resized_mask(mask_path)[0]
            values = np.bincount(mask[mask != 255].ravel().astype(np.int64), minlength=num_classes)
            if values.size > num_classes:
                raise ValueError(f"정답 마스크 클래스 범위 오류: {mask_path}")
            counts += values
        return counts.tolist()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]

        # 이미지 로드 — 채널 수에 따라 RGB 또는 Grayscale 변환
        image = Image.fromarray(read_image(img_path, self.in_channels))
        mask_array, mask_size = self._resized_mask(mask_path)
        if mask_size is not None and image.size != mask_size:
            raise ValueError(f"이미지/마스크 원본 크기 불일치: {img_path} {image.size}, {mask_path} {mask_size}")

        # 리사이즈 (이미지: 픽셀 중심 기준 bilinear, 마스크: 픽셀 중심 기준 nearest
        # + 축소로 사라지는 작은 결함 보존). resize_label_mask 설명 참고.
        image = Image.fromarray(resize(np.asarray(image), self.input_size))
        return self._augment_to_tensors(image, Image.fromarray(mask_array), mask_path)

    def _augment_to_tensors(self, image: Image.Image, mask: Image.Image, mask_path) -> Tuple[torch.Tensor, torch.Tensor]:
        """학습 증강(이미지·마스크 동시) → 정규화 텐서와 클래스 인덱스 텐서. 전체 이미지·패치 학습이 같이 쓴다."""
        # 학습 시 동기화된 데이터 증강
        if self.is_train:
            # 수평 뒤집기 (이미지와 마스크 동시)
            if random.random() < self.flip_prob:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if self.rotation > 0:
                angle = random.uniform(-self.rotation, self.rotation)
                image = T.functional.rotate(
                    image, angle, interpolation=T.InterpolationMode.BILINEAR, fill=0
                )
                mask = T.functional.rotate(
                    mask, angle, interpolation=T.InterpolationMode.NEAREST, fill=255
                )
            if self.color_jitter > 0:
                jitter = T.ColorJitter(
                    brightness=self.color_jitter, contrast=self.color_jitter,
                    saturation=0 if self.in_channels == 1 else self.color_jitter,
                    hue=0 if self.in_channels == 1 else self.color_jitter / 2,
                )
                image = jitter(image)

        # 텐서 변환
        image = T.functional.to_tensor(image)  # (C, H, W), [0, 1]
        image = T.functional.normalize(image, self.mean, self.std)

        # 마스크: (H, W) long 텐서, 값 = 클래스 인덱스
        mask_array = np.array(mask, dtype=np.int64)
        valid = mask_array[mask_array != 255]
        if valid.size and (valid.min() < 0 or
                           (self.num_classes is not None and valid.max() >= self.num_classes)):
            raise ValueError(f"정답 마스크 클래스 범위 오류: {mask_path}")
        mask = torch.from_numpy(mask_array)
        return image, mask


def scaled_size(width: int, height: int, scale: float) -> Tuple[int, int]:
    """패치 학습 배율을 적용한 (가로, 세로). 학습 데이터와 추론이 같은 크기를 쓰도록 한 곳에서 정한다."""
    if not 0 < scale <= 1:
        raise ValueError("패치 학습 배율은 0보다 크고 1 이하여야 합니다")
    if scale == 1:
        return int(width), int(height)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


# tiling 규칙에 적을 수 있는 보간 (OpenCV 플래그 이름). LINEAR_EXACT는 이 키가 없던 9.33 첫 빌드 모델용으로만 남긴다.
SCALE_INTERPOLATIONS = ("INTER_AREA", "INTER_LINEAR_EXACT")


def scale_image(image: np.ndarray, size: Tuple[int, int], interpolation: str = PATCH_SCALE_INTERPOLATION):
    """패치 배율로 판정 영역을 (세로, 가로) 크기로 줄인다. 학습 데이터와 추론이 이 함수 하나를 쓴다.

        1px 크랙, 25%로 축소 (크랙 위치 0~3 = 4픽셀 묶음 안 자리)
            INTER_LINEAR_EXACT  남는 대비 0 · 50 · 50 · 0 %  → 절반 위치에서 사라짐 (점선이 됨)
            INTER_AREA          남는 대비 25 · 25 · 25 · 25 % → 어디서나 옅은 선으로 남음
    """
    import cv2
    if interpolation not in SCALE_INTERPOLATIONS:
        raise ValueError(f"미지원 패치 배율 보간: {interpolation!r}")
    height, width = size
    return cv2.resize(np.asarray(image), (int(width), int(height)), interpolation=getattr(cv2, interpolation))


def patch_origins(length: int, patch: int, stride: int) -> List[int]:
    """한 축을 덮는 패치 시작 위치. 마지막 패치는 끝에 맞춰 이미지 밖으로 나가지 않는다 (짧으면 [0])."""
    if length <= patch:
        return [0]
    origins = list(range(0, length - patch + 1, max(1, stride)))
    if origins[-1] != length - patch:
        origins.append(length - patch)
    return origins


class PatchSegmentationDataset(SegmentationDataset):
    """원본 해상도에서 패치를 잘라 학습하는 분할 데이터셋 (큰 이미지 · 작은 결함 · 적은 장수용).

    전체 이미지를 입력 크기로 줄이면 3500×1504 → 512에서 결함이 7배 작아져 1~2px 결함이 사라진다.
    패치 학습은 줄이지 않고 잘라서 본다.

        학습: 이미지마다 patches_per_image개 (epoch마다 새 위치)
              ├─ defect_ratio 확률: 결함 픽셀 하나를 골라 그 픽셀이 패치 안 임의 위치에 오도록 자름
              └─ 나머지: 이미지 안 아무 위치 (배경 · 결함 없는 이미지 → 오검출 억제)
        검증: 이미지 전체를 겹치지 않게 덮는 고정 타일 (매 epoch 같은 점수)

    패치가 이미지보다 크면 남는 부분은 이미지 0 / 마스크 255(학습 제외)로 채운다.
    디코딩한 이미지·마스크는 작업자 프로세스마다 메모리 한도(기본 1.5GB) 안에서 재사용한다.
    """

    def __init__(self, img_dir: str, mask_dir: str, *, patch_size: int, patches_per_image: int = 0,
                 defect_ratio: float = 0.5, cache_limit_mb: int = 1536, scale: float = 1.0,
                 interpolation: str = PATCH_SCALE_INTERPOLATION, **kwargs):
        if patch_size < 32:
            raise ValueError("패치 크기는 32px 이상이어야 합니다")
        scaled_size(1, 1, scale)  # 배율 범위 확인
        kwargs["input_size"] = (patch_size, patch_size)
        super().__init__(img_dir, mask_dir, **kwargs)
        if not 0 <= defect_ratio <= 1:
            raise ValueError("결함 중심 패치 비율은 0~1이어야 합니다")
        self.patch_size = int(patch_size)
        self.defect_ratio = float(defect_ratio)
        # 배율 < 1: 이미지를 먼저 그만큼 줄인 뒤 패치를 자른다 (추론 계산량 ∝ 배율²).
        # 이미지는 모든 픽셀을 평균하는 축소(scale_image), 마스크는 작은 결함을 보존하는 축소(resize_label_mask).
        self.scale = float(scale)
        if interpolation not in SCALE_INTERPOLATIONS:
            raise ValueError(f"미지원 패치 배율 보간: {interpolation!r}")
        self.interpolation = interpolation
        self._native = OrderedDict()           # 경로 → (이미지, 마스크, 결함 픽셀 위치 표본)
        self._native_bytes = 0
        self._native_limit = int(cache_limit_mb) * 1024 * 1024
        self._sizes = [scaled_size(*self._image_size(image), self.scale) for image, _mask in self.pairs]
        if patches_per_image <= 0:
            # 자동: 이미지 면적 ÷ 패치 면적 (3500×1504, 512 → 이미지당 20개 = 한 epoch에 이미지 전체를 한 번 본 셈)
            area = max(w * h for w, h in self._sizes)
            patches_per_image = max(1, min(64, int(np.ceil(area / (patch_size * patch_size)))))
        self.patches_per_image = int(patches_per_image)
        if not self.is_train:
            # 검증은 겹치지 않는 고정 타일 (마지막 타일만 끝에 맞춰 조금 겹친다)
            self.tiles = [(index, x, y) for index, (w, h) in enumerate(self._sizes)
                          for y in patch_origins(h, patch_size, patch_size)
                          for x in patch_origins(w, patch_size, patch_size)]
        print(f"   패치 학습: {patch_size}×{patch_size}"
              + (f", 배율 {self.scale:.0%}" if self.scale != 1 else ", 원본 해상도") + ", "
              + (f"이미지당 {self.patches_per_image}개 (결함 중심 {self.defect_ratio:.0%})" if self.is_train
                 else f"검증 타일 {len(self.tiles)}개"))

    @staticmethod
    def _image_size(path):
        with Image.open(path) as image:
            return image.size

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_image if self.is_train else len(self.tiles)

    def _load_native(self, index: int):
        """원본 해상도 이미지·마스크(없으면 전부 배경)와 결함 픽셀 위치 표본(최대 2만 개)."""
        img_path, mask_path = self.pairs[index]
        cached = self._native.get(img_path)
        if cached is not None:
            self._native.move_to_end(img_path)
            return cached
        image = read_image(img_path, self.in_channels)
        width, height = image.shape[1], image.shape[0]
        if mask_path is None:
            mask = np.zeros((height, width), np.uint8)
        else:
            with Image.open(mask_path) as source:
                if source.mode not in ("P", "L", "I", "I;16"):
                    raise ValueError(f"정답 마스크는 클래스 인덱스 이미지 필요 (팔레트/L/I): {mask_path}")
                if source.size != (width, height):
                    raise ValueError(f"이미지/마스크 원본 크기 불일치: {img_path} {(width, height)}, "
                                     f"{mask_path} {source.size}")
                mask = np.asarray(source)
                mask = mask.astype(np.uint8) if mask.max(initial=0) <= 255 else mask.astype(np.uint16)
        if self.scale != 1:
            target_w, target_h = scaled_size(width, height, self.scale)
            image = scale_image(image, (target_h, target_w), self.interpolation)  # 1px 크랙도 옅은 선으로 남는다
            mask = resize_label_mask(mask, (target_h, target_w))  # 얇은 흠집·작은 점이 사라지지 않게
        foreground = np.flatnonzero((mask > 0) & (mask != 255))
        if foreground.size > 20000:
            foreground = foreground[np.random.default_rng(index).choice(foreground.size, 20000, replace=False)]
        entry = (image, mask, foreground.astype(np.int64))
        size = image.nbytes + mask.nbytes + entry[2].nbytes
        if size <= self._native_limit:
            while self._native and self._native_bytes + size > self._native_limit:
                _, dropped = self._native.popitem(last=False)
                self._native_bytes -= sum(part.nbytes for part in dropped)
            self._native[img_path] = entry
            self._native_bytes += size
        return entry

    def _origin(self, width, height, foreground):
        """학습 패치 왼쪽 위. 결함 중심이면 결함 픽셀이 패치 안 임의 위치에 오게 (항상 가운데면 위치를 외운다)."""
        patch = self.patch_size
        if foreground.size and random.random() < self.defect_ratio:
            pixel = int(foreground[random.randrange(foreground.size)])
            y, x = divmod(pixel, width)
            left = x - random.randrange(patch)
            top = y - random.randrange(patch)
        else:
            left = random.randrange(max(1, width - patch + 1))
            top = random.randrange(max(1, height - patch + 1))
        return (min(max(0, left), max(0, width - patch)), min(max(0, top), max(0, height - patch)))

    def _crop(self, image, mask, left, top):
        """패치 하나. 이미지보다 크면 이미지 0, 마스크 255(학습 제외)로 채운다."""
        patch = self.patch_size
        piece_image = image[top:top + patch, left:left + patch]
        piece_mask = mask[top:top + patch, left:left + patch]
        if piece_mask.shape != (patch, patch):
            padded_image = np.zeros((patch, patch) + image.shape[2:], image.dtype)
            padded_mask = np.full((patch, patch), 255, mask.dtype)
            padded_image[:piece_image.shape[0], :piece_image.shape[1]] = piece_image
            padded_mask[:piece_mask.shape[0], :piece_mask.shape[1]] = piece_mask
            piece_image, piece_mask = padded_image, padded_mask
        return np.ascontiguousarray(piece_image), np.ascontiguousarray(piece_mask)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_train:
            index = idx // self.patches_per_image
            image, mask, foreground = self._load_native(index)
            left, top = self._origin(image.shape[1], image.shape[0], foreground)
        else:
            index, left, top = self.tiles[idx]
            image, mask, _foreground = self._load_native(index)
        piece_image, piece_mask = self._crop(image, mask, left, top)
        return self._augment_to_tensors(Image.fromarray(piece_image), Image.fromarray(piece_mask),
                                        self.pairs[index][1])

    def class_pixel_counts(self, num_classes: int) -> List[int]:
        """원본 해상도 마스크의 클래스별 픽셀 수 (패치 학습은 줄이지 않으므로 원본 기준)."""
        counts = np.zeros(num_classes, dtype=np.int64)
        for index in range(len(self.pairs)):
            mask = self._load_native(index)[1]
            values = np.bincount(mask[mask != 255].ravel().astype(np.int64), minlength=num_classes)
            if values.size > num_classes:
                raise ValueError(f"정답 마스크 클래스 범위 오류: {self.pairs[index][1]}")
            counts += values
        return counts.tolist()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Anomaly Detection 데이터셋
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AnomalyDataset(Dataset):
    """
    이상 탐지 데이터셋 — 정상(good) 이미지만 학습

    재구성 기반 이상 탐지:
    - 학습: 정상 이미지 입력 → 동일 이미지 재구성이 목표
    - 추론: 재구성 오차가 크면 이상 판정

    폴더 구조:
        train/good/      ← 정상 이미지 (학습용)
        test/good/       ← 정상 이미지 (테스트)
        test/defect/     ← 이상 이미지 (테스트)

    Args:
        root: 데이터 루트 (train/ 또는 test/)
        input_size: 리사이즈 크기
        is_train: 학습 모드
    """
    SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    def __init__(self, root: str,
                 input_size: Tuple[int, int] = (224, 224),
                 in_channels: int = 3,
                 is_train: bool = True, flip_prob: float = 0.5):
        super().__init__()
        self.root = root
        self.input_size = input_size
        self.in_channels = in_channels
        self.is_train = is_train
        self.flip_prob = flip_prob

        # 채널 수에 맞는 정규화 값 선택
        self.mean, self.std = get_normalize_params(in_channels)

        # 이미지 목록 수집 (하위 폴더 재귀 탐색)
        self.samples = []
        self.labels = []  # 0=good, 1=defect
        self._scan_directory(root)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"이미지가 없습니다: {root}\n"
                f"train/good/ 또는 test/{{good,defect}}/ 구조를 확인하세요."
            )

        print(f"Anomaly 데이터셋 로드: {root}")
        print(f"   이미지 수: {len(self.samples)}")
        good_count = sum(1 for l in self.labels if l == 0)
        defect_count = sum(1 for l in self.labels if l == 1)
        print(f"   Good: {good_count} | Defect: {defect_count}")

    def _scan_directory(self, root):
        """하위 폴더 재귀 탐색하여 이미지 수집"""
        for dirpath, dirnames, filenames in os.walk(root):
            # 폴더명으로 라벨 결정
            folder_name = os.path.basename(dirpath).lower()
            label = 1 if "defect" in folder_name or "ng" in folder_name else 0

            for fname in sorted(filenames):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.SUPPORTED_EXT:
                    self.samples.append(os.path.join(dirpath, fname))
                    self.labels.append(label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.samples[idx]
        label = self.labels[idx]

        # 이미지 로드 — 채널 수에 따라 RGB 또는 Grayscale 변환
        image = Image.fromarray(read_image(img_path, self.in_channels))
        image = Image.fromarray(resize(np.asarray(image), self.input_size))

        # 학습 시 약간의 증강 (너무 강하면 재구성이 어려워짐)
        if self.is_train:
            if random.random() < self.flip_prob:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 텐서 변환 (정규화 안함 — 재구성 타겟은 [0,1] 범위)
        image_tensor = T.functional.to_tensor(image)  # (C, H, W), [0, 1]

        # 재구성 기반: 입력 = 타겟 (정규화된 입력, 원본 타겟)
        input_normalized = T.functional.normalize(
            image_tensor.clone(), self.mean, self.std
        )

        # 학습: (normalized_input, original_image)
        # 테스트: (normalized_input, label)
        if self.is_train:
            return input_normalized, image_tensor
        else:
            return input_normalized, torch.tensor(label, dtype=torch.long)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Detection 데이터셋 (정규화 좌표 포맷)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DetectionDataset(Dataset):
    """
    정규화 좌표 포맷 객체 탐지 데이터셋

    레이블 파일 형식 (각 줄):
        class_id  x_center  y_center  width  height
        (모두 0~1 정규화 좌표)

    폴더 구조:
        images/{train,val,test}/  ← 이미지
        labels/{train,val,test}/  ← 레이블 (.txt)
        파일명 매칭: img001.jpg ↔ img001.txt

    Args:
        img_dir: 이미지 디렉토리
        label_dir: 레이블 디렉토리
        input_size: 리사이즈 크기
        num_classes: 클래스 수
        is_train: 학습 모드
        max_objects: 이미지 당 최대 객체 수
    """
    SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    def __init__(self, img_dir: str, label_dir: str,
                 input_size: Tuple[int, int] = (416, 416),
                 num_classes: int = 20,
                 in_channels: int = 3,
                 is_train: bool = True,
                 max_objects: int = 50, flip_prob: float = 0.5):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.is_train = is_train
        self.max_objects = max_objects
        self.flip_prob = flip_prob

        # 채널 수에 맞는 정규화 값 선택
        self.mean, self.std = get_normalize_params(in_channels)

        # 이미지-레이블 쌍 매칭
        self.samples = self._find_pairs()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"이미지-레이블 쌍을 찾을 수 없습니다.\n"
                f"  이미지: {img_dir}\n"
                f"  레이블: {label_dir}\n"
                f"  레이블 형식: class_id x_center y_center width height"
            )

        print("Detection 데이터셋 로드")
        print(f"   이미지-레이블 쌍: {len(self.samples)}")

    def _find_pairs(self) -> List[Tuple[str, str]]:
        from pathlib import Path
        from spatial_data import image_files
        image_root = Path(self.img_dir)
        return [(str(path), str(Path(self.label_dir) / path.relative_to(image_root).with_suffix(".txt")))
                for path in image_files(image_root)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label_path = self.samples[idx]

        # 이미지 로드 & 리사이즈 — 채널 수에 따라 RGB 또는 Grayscale
        image = Image.fromarray(read_image(img_path, self.in_channels))
        image = Image.fromarray(resize(np.asarray(image), self.input_size))

        # 학습 시 데이터 증강
        flip = False
        if self.is_train and random.random() < self.flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flip = True

        # 텐서 변환 + 정규화
        image = T.functional.to_tensor(image)
        image = T.functional.normalize(image, self.mean, self.std)

        from spatial_data import read_spatial_labels
        rows = read_spatial_labels(label_path, "detect", self.num_classes)
        if len(rows) > self.max_objects:
            raise ValueError(f"객체 수 {len(rows)}개가 max_objects={self.max_objects} 초과: {label_path}")
        targets = torch.zeros(self.max_objects, 5)
        if rows:
            labels = torch.tensor(rows, dtype=torch.float32)
            if flip:
                labels[:, 1] = 1.0 - labels[:, 1]
            targets[:len(rows)] = labels

        return image, targets


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  데이터 로더 팩토리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _has_classification_images(root: str, extensions: set) -> bool:
    """Return whether an ImageFolder-style split contains at least one image.

    New projects create empty ``train`` and ``val`` class directories up front.
    An empty val directory must not disable the automatic train/validation split.
    """
    if not os.path.isdir(root):
        return False
    for class_name in os.listdir(root):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            continue
        if any(os.path.isfile(os.path.join(class_dir, filename))
               and os.path.splitext(filename)[1].lower() in extensions
               for filename in os.listdir(class_dir)):
            return True
    return False

def _loader_runtime_options(num_workers: int, pin_memory: bool) -> dict:
    """Options that keep GPU input workers alive and overlap batch preparation."""
    workers = max(0, int(num_workers))
    options = {"num_workers": workers, "pin_memory": bool(pin_memory)}
    if workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    return options


def create_classification_loaders(
    data_root: str,
    input_size: Tuple[int, int] = (224, 224),
    batch_size: int = 8,
    num_workers: int = 0,
    in_channels: int = 3,
    val_split: float = 0.2,
    seed: int = 0,
    pin_memory: bool = False,
    **aug_kwargs
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Classification 데이터 로더 생성

    데이터 구조에 따라 두 가지 모드:
    1. train/val 하위 폴더가 있으면 → 각각 사용
    2. 없으면 → 전체를 val_split 비율로 분할

    Args:
        in_channels: 입력 채널 수 (1=Grayscale, 3=RGB)

    Returns:
        train_loader, val_loader, class_names
    """
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    use_explicit_validation = (
        os.path.isdir(train_dir)
        and _has_classification_images(val_dir, ClassificationDataset.SUPPORTED_EXT)
    )
    if use_explicit_validation:
        # val에 이미지가 있는 경우에만 사용자가 명시적으로 분리한 세트다.
        train_transform = get_classification_transforms(
            input_size, is_train=True, in_channels=in_channels, **aug_kwargs
        )
        val_transform = get_classification_transforms(
            input_size, is_train=False, in_channels=in_channels
        )
        train_dataset = ClassificationDataset(
            train_dir, train_transform, in_channels=in_channels
        )
        val_dataset = ClassificationDataset(
            val_dir, val_transform, in_channels=in_channels,
            class_to_idx=train_dataset.class_to_idx,
        )
    else:
        # ┌──────────────────────────────────────────────────────────┐
        # │ 전체 데이터 또는 빈 val을 train/val 분할하는 경우           │
        # │                                                          │
        # │ random_split은 Subset을 반환하므로 transform을 직접       │
        # │ 바꿀 수 없다. val에 학습 증강(flip, rotation, jitter)이   │
        # │ 적용되면 검증 지표가 비결정적이 되어 모델 선택이 불안정.    │
        # │                                                          │
        # │ 해결: _TransformOverrideSubset 래퍼로 val subset의        │
        # │       transform만 증강 없는 버전으로 교체한다.             │
        # └──────────────────────────────────────────────────────────┘
        train_transform = get_classification_transforms(
            input_size, is_train=True, in_channels=in_channels, **aug_kwargs
        )
        val_transform = get_classification_transforms(
            input_size, is_train=False, in_channels=in_channels
        )
        full_dataset = ClassificationDataset(
            train_dir if os.path.isdir(train_dir) else data_root,
            train_transform, in_channels=in_channels
        )

        # 분할
        total = len(full_dataset)
        if total < 2 or not 0 < val_split < 1:
            raise ValueError("자동 분할에는 이미지 2장 이상과 0~1 사이 검증 비율 필요")
        val_size = max(1, min(total - 1, int(total * val_split)))
        train_size = total - val_size
        train_subset, val_subset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        train_dataset = train_subset
        # val subset: 증강 없는 transform으로 교체
        val_dataset = _TransformOverrideSubset(val_subset, val_transform)

    class_names = (train_dataset.classes
                   if hasattr(train_dataset, 'classes')
                   else train_dataset.dataset.classes)

    loader_options = _loader_runtime_options(num_workers, pin_memory)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_options,
        drop_last=False    # 작은 데이터셋도 실제 학습 배치를 유지
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_options,
    )

    return train_loader, val_loader, class_names


def unlabeled_message(*loaders) -> str:
    """배경으로 학습하는 마스크 없는 이미지 안내 문구. 없으면 빈 문자열."""
    counts = [(name, len(getattr(getattr(loader, "dataset", None), "unlabeled", []) or []))
              for name, loader in zip(("train", "val"), loaders)]
    if not any(count for _, count in counts):
        return ""
    detail = ", ".join(f"{name} {count}장" for name, count in counts if count)
    return (f"마스크 없는 이미지({detail})는 결함이 없는 이미지로 보고 전부 배경으로 학습합니다. "
            "결함이 있는데 아직 그리지 않은 이미지가 있으면 그 결함을 놓치도록 학습되니 먼저 표시하세요.")


def create_segmentation_loaders(
    data_root: str,
    input_size: Tuple[int, int] = (320, 320),
    batch_size: int = 4,
    num_workers: int = 0,
    in_channels: int = 3,
    flip_prob: float = 0.5,
    rotation: float = 0.0,
    color_jitter: float = 0.0,
    num_classes: Optional[int] = None,
    pin_memory: bool = False,
    patch_size: int = 0,
    patches_per_image: int = 0,
    patch_scale: float = 1.0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Segmentation 데이터 로더 생성

    patch_size > 0이면 패치 학습(PatchSegmentationDataset). input_size는 무시하고 패치 크기를 쓴다.
    patch_scale < 1이면 이미지를 그 배율로 줄인 뒤 자른다 (1 = 원본 해상도).

    필요 구조:
        data_root/images/train/, data_root/images/val/
        data_root/masks/train/,  data_root/masks/val/

    Args:
        in_channels: 입력 채널 수 (1=Grayscale, 3=RGB)

    Returns:
        train_loader, val_loader
    """
    if patch_size:
        common = {"patch_size": int(patch_size), "patches_per_image": int(patches_per_image),
                  "scale": float(patch_scale)}
        dataset_type = PatchSegmentationDataset
    else:
        common = {"input_size": input_size}
        dataset_type = SegmentationDataset
    train_dataset = dataset_type(
        img_dir=os.path.join(data_root, "images", "train"),
        mask_dir=os.path.join(data_root, "masks", "train"),
        in_channels=in_channels,
        is_train=True,
        flip_prob=flip_prob,
        rotation=rotation,
        color_jitter=color_jitter,
        num_classes=num_classes,
        **common,
    )
    val_dataset = dataset_type(
        img_dir=os.path.join(data_root, "images", "val"),
        mask_dir=os.path.join(data_root, "masks", "val"),
        in_channels=in_channels,
        is_train=False,
        num_classes=num_classes,
        **common,
    )

    loader_options = _loader_runtime_options(num_workers, pin_memory)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_options,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_options,
    )

    return train_loader, val_loader


def create_anomaly_loaders(
    data_root: str,
    input_size: Tuple[int, int] = (224, 224),
    batch_size: int = 16,
    num_workers: int = 0,
    in_channels: int = 3,
    val_split: float = 0.2,
    flip_prob: float = 0.5,
    seed: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Anomaly Detection 데이터 로더 생성

    학습: data_root/train/good/ 의 정상 이미지
    검증: 학습 데이터에서 val_split 비율만큼 분할

    Args:
        in_channels: 입력 채널 수 (1=Grayscale, 3=RGB)

    Returns:
        train_loader, val_loader
    """
    train_dir = os.path.join(data_root, "train")
    good_dir = os.path.join(train_dir, "good")
    if not os.path.isdir(good_dir):
        raise ValueError("정상 학습 이미지 경로 필요: train/good/")

    full_dataset = AnomalyDataset(
        root=good_dir,
        input_size=input_size,
        in_channels=in_channels,
        is_train=True,
        flip_prob=flip_prob,
    )

    # 학습/검증 분할
    total = len(full_dataset)
    if total < 2 or not 0 < val_split < 1:
        raise ValueError("정상 검증 분할에는 이미지 2장 이상과 0~1 사이 검증 비율 필요")
    val_size = max(1, min(total - 1, int(total * val_split)))
    train_size = total - val_size
    train_dataset, val_subset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    # 반환값은 재구성 타겟 쌍을 유지하되 검증 증강만 제거한다.
    val_base = AnomalyDataset(good_dir, input_size=input_size,
                              in_channels=in_channels, is_train=True,
                              flip_prob=0.0)
    val_dataset = torch.utils.data.Subset(val_base, val_subset.indices)

    loader_options = _loader_runtime_options(num_workers, pin_memory)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_options,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_options,
    )

    return train_loader, val_loader


def create_detection_loaders(
    data_root: str,
    input_size: Tuple[int, int] = (416, 416),
    batch_size: int = 8,
    num_workers: int = 0,
    num_classes: int = 20,
    in_channels: int = 3,
    max_objects: int = 50,
    flip_prob: float = 0.5,
    pin_memory: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Detection 데이터 로더 생성 (정규화 좌표 포맷)

    필요 구조:
        data_root/images/{train,val}/
        data_root/labels/{train,val}/

    Args:
        in_channels: 입력 채널 수 (1=Grayscale, 3=RGB)

    Returns:
        train_loader, val_loader
    """
    from spatial_data import validate_spatial_dataset
    validate_spatial_dataset(data_root, "detect", num_classes)
    train_dataset = DetectionDataset(
        img_dir=os.path.join(data_root, "images", "train"),
        label_dir=os.path.join(data_root, "labels", "train"),
        input_size=input_size,
        num_classes=num_classes,
        in_channels=in_channels,
        is_train=True,
        max_objects=max_objects,
        flip_prob=flip_prob,
    )

    val_img_dir = os.path.join(data_root, "images", "val")
    val_label_dir = os.path.join(data_root, "labels", "val")

    val_loader = None
    if os.path.isdir(val_img_dir):
        try:
            val_dataset = DetectionDataset(
                img_dir=val_img_dir,
                label_dir=val_label_dir,
                input_size=input_size,
                num_classes=num_classes,
                in_channels=in_channels,
                is_train=False,
                max_objects=max_objects,
            )
            loader_options = _loader_runtime_options(num_workers, pin_memory)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                **loader_options,
            )
        except RuntimeError:
            pass  # val 데이터 없으면 None

    loader_options = _loader_runtime_options(num_workers, pin_memory)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_options,
        drop_last=False,
    )

    return train_loader, val_loader
