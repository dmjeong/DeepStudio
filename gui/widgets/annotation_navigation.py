"""Shared save-before-navigation controls for desktop annotation dialogs."""
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QLabel


def hand_off_navigation(dialog, result):
    """이전·다음 이동으로 닫히는 창이 숨기 *전에* 다음 이미지 창을 띄우게 한다.

    이미지마다 창을 새로 만드는 구조라, 닫은 뒤 만들면 그 사이 창이 사라지고 새 창이 기본 크기·위치로
    열려 "껐다 켜지는" 것처럼 보였다. 데이터셋 화면이 넣어 준 navigation_handoff가 다음 창을 이 창과
    같은 자리·크기·최대화 상태로 먼저 보여 주고, 그다음 이 창이 숨는다.

        [다음] → done() ─▶ handoff: 다음 창 생성·같은 geometry로 표시 ─▶ 이 창 숨김
    """
    handoff = getattr(dialog, "navigation_handoff", None)
    if getattr(result, "value", result) != 1 or not getattr(dialog, "navigation_delta", 0) or not callable(handoff):
        return
    dialog.navigation_handoff = None  # 한 번만
    handoff(dialog)


class AnnotationNavigation:
    def done(self, result):
        hand_off_navigation(self, result)
        super().done(result)

    def _add_navigation(self, layout, navigation):
        self.navigation_delta = 0
        if navigation is None:
            return
        index, count = navigation
        controls = QHBoxLayout()
        previous = QPushButton("저장하고 이전  J")
        following = QPushButton("저장하고 다음  K")
        previous.setEnabled(index > 0)
        following.setEnabled(index + 1 < count)
        previous.clicked.connect(lambda: self._navigate(-1))
        following.clicked.connect(lambda: self._navigate(1))
        controls.addWidget(previous)
        controls.addWidget(QLabel(f"{index + 1} / {count}"))
        controls.addWidget(following)
        layout.addLayout(controls)
        self.previous_button, self.next_button = previous, following

    def _navigate(self, delta):
        button = getattr(self, "previous_button" if delta < 0 else "next_button", None)
        if button is None or not button.isEnabled() or self.worker:
            return
        if (getattr(self, "points", ()) or getattr(self, "pending", False) or
                getattr(self.view, "gesture", None)):
            self.status.setText("그리는 중인 영역을 완료하거나 Esc로 취소하세요.")
            return
        self.navigation_delta = delta
        if self.dirty or getattr(self, "save_empty", False):
            self._save()
        else:
            self.accept()
