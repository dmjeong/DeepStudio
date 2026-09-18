"""간접 로드되는 torchvision 네이티브 연산을 버전별 파일명에 관계없이 포함한다."""
from PyInstaller.utils.hooks import collect_dynamic_libs

# 최신 _C_stable.pyd와 기존 _C.pyd, 이미지 확장의 종속 DLL을 모두 수집한다.
binaries = collect_dynamic_libs(
    "torchvision", search_patterns=["*.dll", "*.pyd", "*.so", "*.so.*", "*.dylib"]
)
module_collection_mode = "pyz+py"
