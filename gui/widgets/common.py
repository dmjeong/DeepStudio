"""
Deep Vision Studio — 공통 위젯 모음

마우스 휠 오조작 방지 위젯:
┌──────────────────────────────────────────────────────────────┐
│  문제: 스크롤 패널 안의 SpinBox/ComboBox 위에서 휠을 굴리면   │
│        스크롤 대신 값이 바뀌어 하이퍼파라미터가 오염됨         │
│                                                              │
│  ┌─ 기존 동작 ────────────┐   ┌─ 개선 동작 ────────────────┐ │
│  │ 휠 ↓                   │   │ 휠 ↓                       │ │
│  │  └─► SpinBox가 소비    │   │  └─► ignore() → 부모로 전파│ │
│  │      에폭 100 → 99 (X) │   │      패널이 스크롤 (O)     │ │
│  └────────────────────────┘   └────────────────────────────┘ │
│                                                              │
│  해결: wheelEvent()를 ignore()로 오버라이드                   │
│        → 값 변경 차단 + 이벤트가 부모 QScrollArea로 전파      │
│        → 값 수정은 키보드 입력 / 화살표 버튼으로만 가능        │
└──────────────────────────────────────────────────────────────┘
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSpinBox, QDoubleSpinBox, QComboBox


class _NoWheelMixin:
    """
    마우스 휠 이벤트를 무시하는 믹스인

    - wheelEvent를 ignore()하여 값 변경을 막고,
      이벤트를 부모 위젯(주로 QScrollArea)으로 전파시킨다.
    - 포커스 정책을 StrongFocus로 낮춰 휠만으로 포커스가
      옮겨가지 않도록 한다 (기본 SpinBox는 WheelFocus).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 휠로는 포커스를 얻지 못하게 (클릭/탭 이동으로만 포커스)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        # 값 변경 차단 → 부모(스크롤 영역)가 스크롤을 처리하도록 전파
        event.ignore()


class NoWheelSpinBox(_NoWheelMixin, QSpinBox):
    """휠로 값이 바뀌지 않는 QSpinBox"""
    pass


class NoWheelDoubleSpinBox(_NoWheelMixin, QDoubleSpinBox):
    """휠로 값이 바뀌지 않는 QDoubleSpinBox"""
    pass


class NoWheelComboBox(_NoWheelMixin, QComboBox):
    """
    휠로 항목이 바뀌지 않는 QComboBox

    추가로 sizeHint 를 항목 길이에서 분리한다.

        기본 동작                          이 클래스
        ─────────────────────────────      ─────────────────────────
        가장 긴 항목 전체 폭을 요구   →     최소 N글자 폭만 요구
        → 부모 패널이 함께 넓어지고        → 패널 폭은 레이아웃이
          스플리터 밖으로 잘려나감            자유롭게 결정, 텍스트는
                                             필요 시 말줄임 처리
    """

    #: sizeHint 산정에 쓰는 최소 글자 수 (실제 항목 길이와 무관)
    MIN_CONTENTS_LENGTH = 18

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.setMinimumContentsLength(self.MIN_CONTENTS_LENGTH)
