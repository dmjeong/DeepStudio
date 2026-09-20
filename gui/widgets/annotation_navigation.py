"""Shared save-before-navigation controls for desktop annotation dialogs."""
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QLabel


class AnnotationNavigation:
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
