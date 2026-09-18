"""검출 박스와 인스턴스 분할 폴리곤 라벨의 공통 검증."""
from pathlib import Path
import numpy as np

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def image_files(root):
    root = Path(root)
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS
                  and not any(part.startswith('.') for part in p.relative_to(root).parts))


def semantic_pairs(image_root, mask_root):
    """동일 상대 경로의 이미지/마스크를 연결하고 누락과 중복을 알린다."""
    def indexed(root):
        result = {}
        for path in image_files(root):
            key = path.relative_to(root).with_suffix('')
            if key in result:
                raise ValueError(f'이미지/마스크 이름 중복: {result[key]}, {path}')
            result[key] = path
        return result
    images, masks = indexed(Path(image_root)), indexed(Path(mask_root))
    for key in images.keys() - masks.keys():
        raise ValueError(f'정답 마스크 누락: {images[key]}')
    for key in masks.keys() - images.keys():
        raise ValueError(f'마스크에 대응하는 이미지 누락: {masks[key]}')
    return [(str(image), str(masks[key])) for key, image in images.items()]


def read_spatial_labels(path, task, num_classes):
    path = Path(path)
    if not path.is_file():
        return []  # 개별 라벨 누락은 배경 이미지로 허용한다.
    rows = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = np.asarray([float(value) for value in line.split()], dtype=np.float64)
            if not np.isfinite(row).all() or not len(row):
                raise ValueError('유한한 숫자 필요')
            if row[0] != int(row[0]) or not 0 <= row[0] < num_classes:
                raise ValueError('클래스 번호 범위 오류')
            if task == 'detect':
                if len(row) != 5:
                    raise ValueError('검출은 class cx cy width height의 5개 값 필요')
                if np.any(row[1:] < 0) or np.any(row[1:] > 1) or np.any(row[3:] <= 0):
                    raise ValueError('좌표는 0~1, 너비와 높이는 양수 필요')
            elif task == 'obb':
                from obb import validate_corners
                validate_corners(row[1:])
            elif task == 'segment':
                if len(row) < 7 or len(row) % 2 != 1:
                    raise ValueError('인스턴스 분할은 class x1 y1 x2 y2 x3 y3 ... 폴리곤 필요')
                points = row[1:].reshape(-1, 2)
                if np.any(points < 0) or np.any(points > 1):
                    raise ValueError('폴리곤 좌표는 0~1 범위 필요')
                area = np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1))
                if abs(area) < 1e-12:
                    raise ValueError('폴리곤 면적이 0')
            else:
                raise ValueError('미지원 라벨 태스크')
        except (ValueError, OverflowError) as exc:
            raise ValueError(f'라벨 오류: {path}:{line_number}: {exc}') from exc
        rows.append(row.tolist())
    return rows


def validate_spatial_dataset(root, task, num_classes):
    root = Path(root)
    if num_classes < 1:
        raise ValueError('검출/분할 클래스 목록 필요')
    result = {}
    seen = set()
    for split in ('train', 'val'):
        image_root, label_root = root / 'images' / split, root / 'labels' / split
        images = image_files(image_root)
        if not images:
            raise ValueError(f'{split} 이미지 없음: {image_root}')
        if not label_root.is_dir():
            hint = (' 픽셀 마스크는 커스텀 Semantic Segmentation에 사용하고, 폴리곤 분할에는 폴리곤 txt 라벨 필요.'
                    if task == 'segment' else '')
            raise ValueError(f'라벨 폴더 없음: {label_root}.{hint}')
        objects, background = 0, 0
        label_paths = set()
        for image in images:
            if task == 'obb':
                from obb import require_unrotated_image
                require_unrotated_image(image)
            if image.resolve() in seen:
                raise ValueError(f'학습/검증 이미지 경로 중복: {image}')
            seen.add(image.resolve())
            label = label_root / image.relative_to(image_root).with_suffix('.txt')
            if label in label_paths:
                raise ValueError(f'같은 라벨에 여러 이미지 연결: {label}')
            label_paths.add(label)
            count = len(read_spatial_labels(label, task, num_classes))
            objects += count
            background += count == 0
        if objects == 0:
            hint = ' 픽셀 마스크는 커스텀 분할을 선택하고 폴리곤 분할에는 폴리곤 txt를 제공하세요.' if task == 'segment' else ''
            raise ValueError(f'{split}에 학습/평가 가능한 객체 라벨 없음.{hint}')
        result[split] = {'images': len(images), 'objects': objects, 'background_images': background}
    return result
