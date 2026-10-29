"""Shared save-before-navigation controls for desktop annotation dialogs."""
import gc
import weakref

from PySide6.QtCore import QCoreApplication, QEvent, QThread, QTimer, Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout


def hand_off_navigation(dialog, result):
    """[이전]/[다음]으로 닫히는 작업실이 숨기 *전에* 다음 이미지의 작업실로 바꾸게 한다.

    작업실 창(AnnotationWindow)이 넣어 준 navigation_handoff가 다음 작업실을 만들어 같은 창 안에 끼운다.
    그다음 이 작업실이 숨는다. 그래서 창 안이 비는 순간이 없다.

        [다음] → done() ─▶ handoff: 다음 작업실 생성 · 창 안에서 교체 ─▶ 이 작업실 숨김 · 정리
    """
    handoff = getattr(dialog, "navigation_handoff", None)
    if getattr(result, "value", result) != 1 or not getattr(dialog, "navigation_delta", 0) or not callable(handoff):
        return
    dialog.navigation_handoff = None  # 한 번만
    handoff(dialog)


def release_annotation_dialog(dialog):
    """닫힌 작업실(또는 작업실 창)을 저장·초안 스레드가 끝난 뒤 바로 삭제한다.

    deleteLater만 하면 Qt가 삭제를 예약한 이벤트 루프 단계에서야 처리한다. 원본 해상도 버퍼(4096²에서
    약 160MB)를 가진 작업실이 이미지마다 쌓이지 않도록, 같은 루프 단계에서 삭제를 바로 처리한다.
    """
    for thread in dialog.findChildren(QThread):
        thread.wait()  # 닫힐 때는 작업이 이미 끝나 있어 즉시 반환된다
    dialog.deleteLater()
    QCoreApplication.sendPostedEvents(dialog, QEvent.Type.DeferredDelete)


class AnnotationWindow(QDialog):
    """[이전]/[다음]에도 닫히지 않는 라벨링 작업실 창. 이미지가 바뀌면 창 안의 작업실만 바꿔 끼운다.

        처음      [다음] → 창 닫기 → 새 창을 기본 크기로 열기             → 창이 사라졌다 다른 자리에 나타남
        9.33 1차  [다음] → 새 창을 같은 자리에 먼저 띄우기 → 이전 창 닫기  → 창이 바뀌며 여전히 깜빡임
                                                                     (Windows 창 열림·닫힘 효과, 작업 표시줄, 창 활성화)
        지금      [다음] → 같은 창 안에서 작업실만 교체                    → 이미지와 라벨만 바뀐다

    작업실(분할·탐지 작업실, OBB 편집)은 이미지마다 새로 만든다. 저장·초안·복구 규칙이 이미지 한 장 단위라서다.
    창(크기·위치·최대화·모달)은 하나로 유지한다. 패널 폭·보기 설정 같은 화면 배치는 다음 작업실이 이어받는다.
    build(위치)는 그 위치 이미지의 작업실(QDialog)을 만들어 돌려준다(아직 보이지 않음).
    """

    def __init__(self, build, index, count, parent=None, *, geometry=None):
        super().__init__(parent)
        self._build = build
        self.index, self.count = index, count
        self.workspace = None
        self.error = ""  # 다음 이미지를 열지 못한 이유 (창을 닫은 뒤 데이터셋 화면이 알린다)
        self._retired = []  # 교체된 이전 작업실 (스레드가 끝나면 삭제)
        # 오래 쓰는 큰 작업 창이라 최대화 단추를 둔다 (대화상자 기본값에는 없다)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.resize(1380, 900)
        if geometry is not None:
            self.restoreGeometry(geometry)  # 지난번 크기·위치·최대화 상태
        self._mount(build(index))

    def _mount(self, workspace):
        """작업실을 이 창의 내용으로 넣는다. 바꾸는 동안 그리기를 멈춰, 이전 화면이 새 화면으로 한 번에 바뀐다."""
        previous, self.workspace = self.workspace, workspace
        workspace.setParent(self, Qt.WindowType.Widget)  # 따로 뜨는 창이 아니라 이 창 안의 위젯
        workspace.navigation_handoff = self._hand_off
        ref = weakref.ref(workspace)  # 연결이 작업실을 붙잡지 않게
        workspace.finished.connect(lambda result: self._finished(ref(), result))
        self.setUpdatesEnabled(False)
        try:
            if previous is not None:
                self.layout().removeWidget(previous)
            self.layout().addWidget(workspace)
            workspace.show()
            if previous is not None:
                if hasattr(workspace, "adopt_view_state"):
                    workspace.adopt_view_state(previous)
                previous.hide()
            self.setWindowTitle(workspace.windowTitle())
            if self.isVisible():  # 첫 작업실은 창이 뜬 뒤 자기 타이머로 맞춘다
                self.layout().activate()
                if workspace.layout() is not None:
                    workspace.layout().activate()
                if hasattr(workspace, "_fit"):
                    workspace._fit()  # 첫 화면부터 화면 맞춤 (확대가 튀지 않게)
                if hasattr(workspace, "center_current_thumbnail"):
                    workspace.center_current_thumbnail()
            view = getattr(workspace, "view", None)
            if view is not None:
                view.setFocus()  # J/K·Space 같은 단축키가 계속 캔버스로 간다
        finally:
            self.setUpdatesEnabled(True)

    def _hand_off(self, current):
        """current가 [이전]/[다음]으로 닫히기 직전(아직 보이는 동안) 다음 이미지의 작업실로 바꾼다."""
        delta = current.navigation_delta
        target = self.index + delta if type(delta) is int else -1
        if not delta or not 0 <= target < self.count:
            return
        if hasattr(current, "release_buffers"):
            current.release_buffers()  # 다음 작업실이 이 메모리를 재사용하도록 먼저 놓는다
        try:
            following = self._build(target)
        except (OSError, ValueError) as exc:
            # 이 작업실이 닫히면 창도 닫히고, 데이터셋 화면이 이유를 알린다 (지금까지의 작업은 이동 전에 저장됐다)
            self.error = ("다음 이미지를 열 수 없어 작업실을 닫았습니다. 지금까지의 작업은 저장(또는 초안으로 보관)됐습니다.\n"
                          + (str(exc) or type(exc).__name__))
            return
        self.index = target
        self._mount(following)

    def _finished(self, workspace, result):
        if workspace is None:
            return
        if workspace is self.workspace:
            # 이어서 열 작업실이 없다(닫기·마지막 이미지·열 수 없는 이미지) → 창을 닫는다.
            # 같은 호출 안에서 닫아, 내용이 빠진 빈 창이 한 번도 그려지지 않게 한다.
            self.done(result)
            return
        # 이미 다음 작업실로 바뀌었다. 이전 작업실은 저장·초안 스레드가 끝난 뒤 지운다.
        self._retired.append(workspace)
        QTimer.singleShot(0, self, self._release_retired)

    def _release_retired(self):
        while self._retired:
            workspace = self._retired.pop()
            release_annotation_dialog(workspace)
            del workspace
        gc.collect()  # 작업실↔캔버스 참조 순환이 잡은 원본 해상도 numpy 버퍼를 바로 회수

    def reject(self):
        """창 닫기(X)·Esc는 지금 작업실의 닫기 규칙을 따른다 (저장 중이면 대기, 미저장 편집은 초안으로 보존).

        작업실이 실제로 닫히면 finished로 이 창도 닫힌다.
        """
        workspace = self.workspace
        if workspace is not None and workspace.isVisibleTo(self):
            workspace.reject()
        else:
            super().reject()

    def closeEvent(self, event):
        event.ignore()
        self.reject()


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
