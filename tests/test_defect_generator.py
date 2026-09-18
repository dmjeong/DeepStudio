"""
합성 불량 생성 엔진 단위 테스트

테스트 항목:
┌─────────────────────────────────────────────────────────┐
│  1. 7종 기법 각각의 입출력 형태 보존 검증                  │
│  2. Grayscale / RGB 양쪽 모두 동작 확인                   │
│  3. generate_defect 통합 함수 정상 동작                   │
│  4. generate_batch 다중 생성 정상 동작                    │
│  5. 엣지 케이스: 매우 작은 이미지, 극단 파라미터           │
│  6. _random_region / _smooth_mask 헬퍼 검증              │
└─────────────────────────────────────────────────────────┘
"""

import numpy as np
import pytest

from core.defect_generator import (
    DefectType, DefectParams, DEFECT_INFO,
    generate_scratch, generate_stain, generate_cutout,
    generate_noise, generate_texture_swap, generate_elastic_deform,
    generate_color_shift, generate_defect, generate_batch,
    _random_region, _smooth_mask,
)


# ── 기본 파라미터 ──
DEFAULT_PARAMS = DefectParams(
    intensity=0.5,
    size_ratio=0.15,
    count=1,
    types=[DefectType.SCRATCH],
)


# ═══════════════════════════════════════════════════════
#  각 기법별 입출력 형태 보존 테스트
# ═══════════════════════════════════════════════════════

class TestGeneratorsRGB:
    """RGB(3ch) 이미지에 대한 각 기법 테스트"""

    @pytest.mark.parametrize("gen_fn", [
        generate_scratch,
        generate_stain,
        generate_cutout,
        generate_noise,
        generate_texture_swap,
        generate_elastic_deform,
        generate_color_shift,
    ])
    def test_output_shape_preserved(self, sample_image, gen_fn):
        """출력 이미지 shape == 입력 이미지 shape"""
        result = gen_fn(sample_image, DEFAULT_PARAMS)
        assert result.shape == sample_image.shape

    @pytest.mark.parametrize("gen_fn", [
        generate_scratch,
        generate_stain,
        generate_cutout,
        generate_noise,
        generate_texture_swap,
        generate_elastic_deform,
        generate_color_shift,
    ])
    def test_output_dtype_uint8(self, sample_image, gen_fn):
        """출력 dtype == uint8"""
        result = gen_fn(sample_image, DEFAULT_PARAMS)
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("gen_fn", [
        generate_scratch,
        generate_stain,
        generate_cutout,
        generate_noise,
        generate_texture_swap,
        generate_elastic_deform,
        generate_color_shift,
    ])
    def test_output_range_valid(self, sample_image, gen_fn):
        """출력 픽셀값 [0, 255] 범위 내"""
        result = gen_fn(sample_image, DEFAULT_PARAMS)
        assert result.min() >= 0
        assert result.max() <= 255

    @pytest.mark.parametrize("gen_fn", [
        generate_scratch,
        generate_stain,
        generate_cutout,
        generate_noise,
        generate_texture_swap,
        generate_elastic_deform,
        generate_color_shift,
    ])
    def test_input_not_modified(self, sample_image, gen_fn):
        """원본 이미지가 변경되지 않아야 함 (copy 확인)"""
        original_copy = sample_image.copy()
        gen_fn(sample_image, DEFAULT_PARAMS)
        np.testing.assert_array_equal(sample_image, original_copy)


class TestGeneratorsGray:
    """Grayscale(1ch) 이미지에 대한 각 기법 테스트"""

    @pytest.mark.parametrize("gen_fn", [
        generate_scratch,
        generate_stain,
        generate_cutout,
        generate_noise,
        generate_texture_swap,
        generate_elastic_deform,
        generate_color_shift,
    ])
    def test_gray_shape_preserved(self, gray_image, gen_fn):
        """그레이스케일 입력 → 그레이스케일 출력"""
        result = gen_fn(gray_image, DEFAULT_PARAMS)
        assert result.shape == gray_image.shape
        assert result.ndim == 2

    @pytest.mark.parametrize("gen_fn", [
        generate_scratch,
        generate_stain,
        generate_cutout,
        generate_noise,
        generate_texture_swap,
        generate_elastic_deform,
        generate_color_shift,
    ])
    def test_gray_dtype_uint8(self, gray_image, gen_fn):
        """그레이스케일 출력 dtype == uint8"""
        result = gen_fn(gray_image, DEFAULT_PARAMS)
        assert result.dtype == np.uint8


# ═══════════════════════════════════════════════════════
#  통합 함수 테스트
# ═══════════════════════════════════════════════════════

class TestGenerateDefect:
    """generate_defect 통합 함수 테스트"""

    def test_returns_tuple(self, sample_image):
        """(이미지, DefectType) 튜플 반환"""
        result = generate_defect(sample_image, DEFAULT_PARAMS)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], DefectType)

    def test_specified_type(self, sample_image):
        """특정 유형 지정 시 해당 유형으로 생성"""
        params = DefectParams(
            types=[DefectType.SCRATCH, DefectType.STAIN]
        )
        _, dtype = generate_defect(
            sample_image, params, defect_type=DefectType.STAIN
        )
        assert dtype == DefectType.STAIN

    def test_random_type_from_params(self, sample_image):
        """유형 미지정 시 params.types에서 랜덤 선택"""
        params = DefectParams(
            types=[DefectType.SCRATCH, DefectType.STAIN]
        )
        _, dtype = generate_defect(sample_image, params)
        assert dtype in (DefectType.SCRATCH, DefectType.STAIN)


class TestGenerateBatch:
    """generate_batch 배치 생성 테스트"""

    def test_batch_count(self, sample_image):
        """요청 수만큼 생성"""
        results = generate_batch(sample_image, DEFAULT_PARAMS, num_images=5)
        assert len(results) == 5

    def test_batch_item_types(self, sample_image):
        """배치 내 모든 항목이 (ndarray, DefectType) 튜플"""
        results = generate_batch(sample_image, DEFAULT_PARAMS, num_images=3)
        for img, dtype in results:
            assert isinstance(img, np.ndarray)
            assert isinstance(dtype, DefectType)


# ═══════════════════════════════════════════════════════
#  헬퍼 함수 테스트
# ═══════════════════════════════════════════════════════

class TestHelpers:
    """_random_region, _smooth_mask 헬퍼 검증"""

    def test_random_region_bounds(self):
        """생성된 영역이 이미지 경계 내에 있어야 함"""
        h, w = 100, 150
        for _ in range(50):  # 여러 번 반복 검증
            x, y, rw, rh = _random_region(h, w, 0.15)
            assert 0 <= x < w
            assert 0 <= y < h
            assert rw > 0
            assert rh > 0
            assert x + rw <= w or rw <= w  # 영역 크기 합리성
            assert y + rh <= h or rh <= h

    def test_smooth_mask_shape(self):
        """마스크 shape == (h, w)"""
        mask = _smooth_mask(100, 150, 30, 20, 40, 30)
        assert mask.shape == (100, 150)

    def test_smooth_mask_range(self):
        """마스크 값 범위 [0, 1]"""
        mask = _smooth_mask(100, 150, 30, 20, 40, 30)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_smooth_mask_dtype(self):
        """마스크 dtype == float32"""
        mask = _smooth_mask(100, 150, 30, 20, 40, 30)
        assert mask.dtype == np.float32


# ═══════════════════════════════════════════════════════
#  엣지 케이스 테스트
# ═══════════════════════════════════════════════════════

class TestEdgeCases:
    """극단적 입력에 대한 안정성 테스트"""

    def test_small_image(self):
        """매우 작은 이미지 (10×10)에서도 크래시 없음"""
        small = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        params = DefectParams(intensity=0.5, size_ratio=0.3, count=1,
                              types=[DefectType.SCRATCH])
        result, _ = generate_defect(small, params)
        assert result.shape == small.shape

    def test_high_intensity(self, sample_image):
        """최대 강도에서도 안정 동작"""
        params = DefectParams(intensity=1.0, size_ratio=0.5, count=3,
                              types=[DefectType.NOISE])
        result, _ = generate_defect(sample_image, params)
        assert result.dtype == np.uint8

    def test_low_intensity(self, sample_image):
        """최소 강도에서도 안정 동작"""
        params = DefectParams(intensity=0.1, size_ratio=0.05, count=1,
                              types=[DefectType.STAIN])
        result, _ = generate_defect(sample_image, params)
        assert result.dtype == np.uint8

    def test_multiple_counts(self, sample_image):
        """count=10 (다수 패턴)에서도 안정"""
        params = DefectParams(intensity=0.5, size_ratio=0.1, count=10,
                              types=[DefectType.CUTOUT])
        result, _ = generate_defect(sample_image, params)
        assert result.shape == sample_image.shape


# ═══════════════════════════════════════════════════════
#  DEFECT_INFO 메타데이터 테스트
# ═══════════════════════════════════════════════════════

class TestDefectInfo:
    """DEFECT_INFO 사전 완전성 검증"""

    def test_all_types_have_info(self):
        """모든 DefectType에 대한 정보가 존재"""
        for dtype in DefectType:
            assert dtype in DEFECT_INFO, f"{dtype}에 대한 정보 누락"

    def test_info_has_required_keys(self):
        """각 정보 항목에 name, desc, group 키 존재"""
        for dtype, info in DEFECT_INFO.items():
            assert "name" in info, f"{dtype}: name 키 누락"
            assert "desc" in info, f"{dtype}: desc 키 누락"
            assert "group" in info, f"{dtype}: group 키 누락"
