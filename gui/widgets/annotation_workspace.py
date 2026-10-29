"""Shared Windows annotation workspace for axis-aligned boxes and semantic masks."""

from __future__ import annotations

import copy
import uuid
from pathlib import Path

import numpy as np
from core.annotation_tools import AnnotationDraftStore, fingerprint
from PySide6.QtCore import QEvent, QObject, QRectF, QSignalBlocker, Qt, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap, QTransform
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QInputDialog,
    QPushButton,
    QTableWidgetItem,
    QToolButton,
    QWidget,
)
from widgets.annotation_grid_dialog import GridDialog
from widgets.annotation_io import AnnotationWrite, ImagePrefetcher
from widgets.annotation_navigation import hand_off_navigation
from widgets.annotation_workspace_state import AnnotationWorkspaceState
from widgets.annotation_workspace_ui import AnnotationWorkspaceUI
from widgets.common import NoWheelComboBox


class SpacePan(QObject):
    """작업실 어디에 포커스가 있든 Space는 캔버스 화면 이동(누른 채 드래그)으로 보낸다.

    Space는 캔버스에 포커스가 있을 때만 화면 이동이었다. 도구 버튼·체크박스·콤보를 한 번 누르면
    포커스가 그쪽으로 가서, Space가 그 버튼을 다시 누르거나 체크를 바꾸거나 목록을 열었다.
    글자를 입력하는 칸(클래스 이름, 숫자 칸, 단축키 입력)은 Space를 그대로 쓴다. 작업실 위에 뜬
    다른 창(격자·단축키 설정)은 건드리지 않는다.
    """

    def __init__(self, workspace):
        super().__init__(workspace)
        self.workspace = workspace

    @staticmethod
    def _text_entry(widget):
        from PySide6.QtWidgets import QAbstractSpinBox, QKeySequenceEdit, QLineEdit, QPlainTextEdit, QTextEdit

        return isinstance(widget, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QKeySequenceEdit))

    def eventFilter(self, watched, event):
        if event.type() not in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            return False
        if event.key() != Qt.Key.Key_Space or not isinstance(watched, QWidget):
            return False
        view = getattr(self.workspace, "view", None)
        if view is None or watched.window() is not self.workspace or self._text_entry(watched):
            return False
        if watched is view or view.isAncestorOf(watched):
            return False  # 캔버스는 원래 처리 (누르면 이동 커서, 떼면 해제)
        if event.type() == QEvent.Type.KeyPress:
            view.setFocus()  # 이후 드래그·단축키가 캔버스로 간다
            view.keyPressEvent(event)
        else:
            view.keyReleaseEvent(event)
        return True  # 포커스를 가진 버튼·체크박스·콤보가 Space로 눌리지 않게 막는다


class AnnotationWorkspace(AnnotationWorkspaceState, AnnotationWorkspaceUI, QDialog):
    """One editing/saving contract for every detection/segmentation model."""

    def __init__(
        self,
        image_path,
        class_names,
        source,
        save,
        parent=None,
        *,
        task,
        add_class=None,
        navigation=None,
        image_paths=None,
        draft_store=None,
        preferences=None,
        label_revision=None,
        image_records=None,
        revision_provider=None,
    ):
        super().__init__(parent)
        self.task = task
        self.image_path = str(Path(image_path).resolve())
        self._image_revision = AnnotationDraftStore.image_revision(self.image_path)
        self.names = list(class_names)
        self.document = source if task == "segment" else None
        if self.document is not None:
            self.document.ensure_object_ids(self.image_path)
            self.document.managed_cache = True
        self.rows = copy.deepcopy(source) if task == "detect" else self.document.shapes
        self.ids = [uuid.uuid4().hex for _ in self.rows] if task == "detect" else []
        self.object_meta = {} if task == "detect" else self.document.objects
        self._save_callback, self._add_class_callback = save, add_class
        self.worker = self.draft_worker = self.ai_worker = None
        self.dirty = False
        self.pending = False
        self.save_empty = not self.rows
        self.selected = -1
        self.selected_indices = set()
        self.undo_stack, self.redo_stack = [], []
        self.revision = 0
        self.navigation_delta = 0
        self.navigation = navigation
        self.image_paths = list(image_paths or [])
        self.preferences = preferences if preferences is not None else {}
        self.label_revision = label_revision
        self.revision_provider = revision_provider
        self.image_records = {
            str(Path(r["path"]).resolve()): dict(r) for r in image_records or []
        }
        self.drafts = draft_store or AnnotationDraftStore()
        self._restorable = None
        self._table_updating = False
        self._closing = False
        self._grid_dialog = None
        self._ai_engine = None
        self._ai_request = 0
        self._ai_proposal = None
        self._save_review = False
        self.reviewed = False
        self._current_thumbnail_state = (
            "정답 있음"
            if self.rows or (self.document is not None and self.document.persisted)
            else "미작업"
        )
        # The previous window decoded this file in the background while the user
        # was editing; without it the GUI thread decodes (4096² PNG ≈ 0.4 s).
        prefetched = ImagePrefetcher.take(self.image_path)
        self.pixmap = (
            QPixmap.fromImage(prefetched)
            if prefetched is not None
            else QPixmap(self.image_path)
        )
        if self.pixmap.isNull():
            raise ValueError("이미지를 열 수 없습니다.")
        self.pixmap.setDevicePixelRatio(1.0)
        self.width_px, self.height_px = self.pixmap.width(), self.pixmap.height()
        self.points = []  # Compatibility with the old two-click box constructor.
        self._baseline_state = self._capture()
        self._baseline = self._label_fingerprint(self._baseline_state)
        self._draft_timer = QTimer(self)
        self._draft_timer.setSingleShot(True)
        self._draft_timer.setInterval(1000)
        self._draft_timer.timeout.connect(self._write_draft)
        self._source_levels_image = None
        self._levels_timer = QTimer(self)
        self._levels_timer.setSingleShot(True)
        self._levels_timer.setInterval(80)
        self._levels_timer.timeout.connect(self._apply_levels)
        self.setWindowTitle(
            ("Object detection" if task == "detect" else "Segmentation")
            + " · 라벨링 작업실 · "
            + Path(image_path).name
        )
        self.resize(1380, 900)
        self.setMinimumSize(620, 340)
        self._build_ui()
        from widgets.annotation_shortcuts import AnnotationShortcuts

        self.shortcuts = AnnotationShortcuts(self)
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
        # 버튼·체크박스·콤보는 클릭해도 키보드 포커스를 가져가지 않는다: Space(이동)·Enter(다각형 완료)·
        # K/J 같은 단축키가 계속 캔버스로 간다. 글자·숫자 입력칸은 그대로 포커스를 받는다.
        for kind in (QPushButton, QToolButton, QCheckBox, QComboBox):
            for widget in self.findChildren(kind):
                if not (isinstance(widget, QComboBox) and widget.isEditable()):
                    widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._space_pan = SpacePan(self)
        QApplication.instance().installEventFilter(self._space_pan)
        self._restore_preferences()
        self._load_draft()
        # 초안 복원은 문서를 렌더링된 새 객체로 바꾸므로, 어느 경우든 캐시가 지금 문서와 일치한다.
        self._render(reuse_cache=True)
        QTimer.singleShot(0, self._fit)
        if self.navigation and self.image_paths:
            current = self.navigation[0]
            ImagePrefetcher.request(
                [
                    self.image_paths[i]
                    for i in (current + 1, current - 1)
                    if 0 <= i < len(self.image_paths)
                ]
            )

    def done(self, result):
        # A closed window's remaining thumbnails must not keep decoding originals
        # on the shared IO threads while the next image's window is opening.
        worker = getattr(self, "thumbnail_worker", None)
        if worker is not None:
            try:
                worker.requestInterruption()
            except RuntimeError:  # finished loader already deleted itself
                pass
        if getattr(self, "_space_pan", None) is not None:
            QApplication.instance().removeEventFilter(self._space_pan)
            self._space_pan = None
        hand_off_navigation(self, result)  # 다음 이미지 창을 먼저 같은 자리에 띄운 뒤 숨는다
        super().done(result)

    def release_buffers(self):
        """곧 숨을 이 창의 원본 해상도 배열(4096²에서 약 160MB)을 다음 이미지 창을 만들기 전에 놓는다.

        이전 창을 닫기 전에 다음 창을 띄우면 두 창이 잠시 함께 산다. 이 배열을 먼저 놓지 않으면 다음 창이
        새 메모리를 받아야 해서(첫 사용 page fault) 창 생성이 0.1초 → 0.3초로 느려졌다. 화면은 이미 그려져
        있고 이 창은 바로 숨으므로 필요 없다. 다시 쓰이면 문서 캐시는 필요할 때 다시 계산된다.
        """
        if self.document is not None:
            self.document.invalidate()
            self.document._pixels = self.document._owners = None
        view = getattr(self, "view", None)
        if view is not None and hasattr(view, "_view_pixels"):
            view._view_pixels = view._view_owners = None

    def _items(self):
        return self.document.shapes if self.task == "segment" else self.rows

    def _set_annotation_visibility(self, visible):
        if self.task == "segment":
            self.view.overlay_visible = bool(visible)
            self.view.overlay.setVisible(bool(visible))
        else:
            self.view.show_annotations = bool(visible)
        self.view.viewport().update()

    def _set_label_visibility(self, visible):
        self.view.labels_visible = bool(visible)
        self.view.viewport().update()

    def _reset_levels(self):
        self.brightness.setValue(0)
        self.contrast.setValue(0)
        self._levels_timer.stop()
        self._apply_levels()

    def _apply_levels(self):
        if self.brightness.value() == 0 and self.contrast.value() == 0:
            self.view.source_item.setPixmap(self.pixmap)
            return
        try:
            if self._source_levels_image is None:
                from PIL import Image

                image = self.pixmap.toImage().convertToFormat(
                    QImage.Format.Format_RGBA8888
                )
                rgba = (
                    np.frombuffer(image.bits(), dtype=np.uint8)
                    .reshape(image.height(), image.bytesPerLine())[
                        :, : image.width() * 4
                    ]
                    .reshape(image.height(), image.width(), 4)
                    .copy()
                )
                self._source_levels_image = Image.fromarray(rgba)
            value = np.arange(256, dtype=float)
            value = (
                (value - 127.5) * (10 ** (self.contrast.value() / 100))
                + 127.5
                + self.brightness.value() * 1.275
            )
            table = np.clip(value, 0, 255).astype(np.uint8).tolist()
            shown = self._source_levels_image.point(table * 3 + list(range(256)))
            raw = shown.tobytes()
            image = QImage(
                raw,
                shown.width,
                shown.height,
                shown.width * 4,
                QImage.Format.Format_RGBA8888,
            ).copy()
            self.view.source_item.setPixmap(QPixmap.fromImage(image))
        except (ValueError, OSError, MemoryError) as exc:
            self.status.setText("표시 조절 실패: " + str(exc))

    def _key(self, index):
        return self.document.key(index) if self.task == "segment" else self.ids[index]

    def _label_fingerprint(self, state=None):
        if self.task == "detect":
            from core.dataset_editor import encode_annotations

            return fingerprint(
                encode_annotations(state["items"] if state else self.rows).decode(
                    "utf-8"
                )
            )
        document = state["document"] if state else self.document
        return fingerprint(
            {
                "width": document.width,
                "height": document.height,
                "classes": document.classes,
                "base": document.base_digest,
                "shapes": document.shapes,
                "objects": document.objects,
            }
        )

    def _changed(self, region=None):
        self.revision += 1
        self.dirty = True
        self.reviewed = False
        self._current_state("작업 중")
        self._draft_timer.start()
        if self.ai_worker or self._ai_proposal is not None:
            self._cancel_ai()
        self._render(region)

    def _create_box(self, coordinates):
        if not self.names:
            return
        self._checkpoint("박스 생성")
        self.rows.append(
            {
                "class_id": max(0, self.class_combo.currentIndex()),
                "coordinates": list(coordinates),
            }
        )
        self.ids.append(uuid.uuid4().hex)
        self.selected = len(self.rows) - 1
        self.selected_indices = {self.selected}
        self._changed()

    def _edit_box(self, index, coordinates):
        self._edit_many({index: coordinates})

    def _create(self, shape):
        shape = dict(shape)
        shape.setdefault(
            "protected_objects",
            [key for key, meta in self.object_meta.items() if meta.get("locked")],
        )
        shape = self.document.validate_shape(shape)
        if shape.get("operation") != "erase" and "object_id" not in shape:
            shape["object_id"] = uuid.uuid4().hex
        self._checkpoint(
            "지우기" if shape.get("operation") == "erase" else "영역 그리기"
        )
        region = self.document.append_shape(shape)
        if shape.get("operation") != "erase":
            self.selected = self.document.representative(shape["object_id"])
            self.selected_indices = {self.selected}
        self._changed(region)
        if shape.get("operation") != "erase":
            # 그린 직후 클래스를 고르면 이 객체에 적용된다 (그리고 → 클래스 고르기).
            self.status.setText("그린 영역이 선택됐습니다 · 클래스를 고르면 이 영역에 적용 · "
                                "다음 영역 클래스만 바꾸려면 Esc로 선택 해제 후 고르세요")

    def _edit(self, index, shape):
        self._edit_many({index: shape})

    def _edit_many(self, changes):
        valid = {
            i: value
            for i, value in changes.items()
            if i >= 0 and not self.object_meta.get(self._key(i), {}).get("locked")
        }
        if not valid:
            return
        if self.task == "segment":
            try:
                valid = {
                    i: self.document.validate_shape(value) for i, value in valid.items()
                }
            except ValueError as exc:
                self.status.setText(str(exc))
                return
        self._checkpoint("영역 이동·크기 수정")
        for index, value in valid.items():
            if self.task == "segment":
                self.document.shapes[index] = value
            else:
                self.rows[index]["coordinates"] = list(value)
        self._changed()

    def _select(self, index):
        self.selected = index
        if index == -1:
            self.selected_indices = set()
        elif index not in self.selected_indices:
            self.selected_indices = {index}
        self.view.selected = index
        if index in getattr(self, "_visible_indices", []):
            self.table.scrollToItem(
                self.table.item(self._visible_indices.index(index), 0)
            )
        self._sync_table_selection()
        self._selection_options()
        self._selection_summary()
        self._buttons()

    def _select_box(self, index):
        self._select_many({index})
        self._select(index)

    def _select_many(self, indices):
        self.selected_indices = set(indices)
        self.view.selected_set = set(indices)
        if self.selected not in indices:
            self.selected = next(iter(sorted(indices)), -1)
        self.view.selected = self.selected
        self._sync_table_selection()
        self._selection_options()
        self._selection_summary()
        self._buttons()
        self.view.viewport().update()

    def _selection_summary(self):
        self.object_summary.setText(
            f"영역 {len(getattr(self, '_visible_indices', []))}개 · 선택 {len(self.selected_indices)}개"
        )

    def _selection_options(self):
        if self.task == "segment":
            # 브러시는 선택한 '브러시 객체'에만 이어 그린다 (한 결함을 여러 획으로 칠하기).
            # 사각형·다각형·AI·분리로 만든 객체가 선택돼 있을 때 이어 그리면 그 객체에 합쳐져서,
            # 맞닿은 브러시 영역을 다시 나눌 수 없었다. 이때는 새 객체로 시작한다.
            chosen = self.selected if self.selected >= 0 else -1
            continue_brush = False
            if chosen >= 0 and self.document.shapes[chosen]["class_id"] == self._active_class_id():
                key = self._key(chosen)
                kinds = {s["kind"] for i, s in enumerate(self.document.shapes)
                         if s.get("operation") != "erase" and self.document.key(i) == key}
                continue_brush = kinds == {"stroke"}
            self.view.brush_object = self._key(chosen) if continue_brush else None

    def _table_selection(self):
        if self._table_updating:
            return
        rows = {item.row() for item in self.table.selectedItems()}
        indices = {
            self._visible_indices[r] for r in rows if r < len(self._visible_indices)
        }
        self._select_many(indices)

    @Slot(int)
    def _table_class_changed(self, value):
        combo = self.sender()
        if combo is not None and value >= 0:
            self._change_class(int(combo.property("index")), int(combo.itemData(value)))

    @Slot(bool)
    def _table_remove_clicked(self, _checked=False):
        button = self.sender()
        if button is not None:
            self._remove(int(button.property("index")))

    @Slot(bool)
    def _table_flag_changed(self, value):
        check = self.sender()
        if check is not None:
            self._set_flag(
                int(check.property("index")),
                check.property("flag"),
                not value if check.property("invert") else value,
            )

    def _sync_table_selection(self):
        self._table_updating = True
        self.table.clearSelection()
        from PySide6.QtCore import QItemSelectionModel

        for row, index in enumerate(getattr(self, "_visible_indices", [])):
            if index in self.selected_indices:
                self.table.selectionModel().select(
                    self.table.model().index(row, 0),
                    QItemSelectionModel.SelectionFlag.Select
                    | QItemSelectionModel.SelectionFlag.Rows,
                )
        self._table_updating = False

    def _indices(self):
        if self.task == "detect":
            return list(range(len(self.rows)))
        return self.view.object_indices()

    def _class_id(self, index):
        return -index - 2 if index < -1 else self._items()[index]["class_id"]

    def _class_picked(self, *_):
        """클래스 목록·콤보·숫자키로 클래스를 고르면 선택한 객체(방금 그린 객체 포함)에 적용한다.

        예전에는 활성(그리기) 클래스만 바뀌고 선택 객체는 그대로라 '선택에 적용'을 따로 눌러야 했다.
        그래서 사각형 두 개를 그리고 클래스를 골라도 둘 다 처음 클래스로 남았다. 다음에 그릴 클래스만
        바꾸려면 Esc로 선택을 해제한 뒤 고른다. 분할의 배경(0)은 적용하지 않는다(삭제로).
        """
        if not self.selected_indices:
            return
        class_id = self._active_class_id()
        indices = {i for i in self.selected_indices if i in self._indices()}
        if not indices or all(self._class_id(i) == class_id for i in indices):
            return
        if self.task == "segment" and class_id == 0:
            self.status.setText("선택 영역을 배경으로 바꾸려면 '선택 삭제'를 쓰세요.")
            return
        self._reclass(indices, class_id)
        name = self.names[class_id] if 0 <= class_id < len(self.names) else str(class_id)
        self.status.setText(f"선택한 영역 {len(indices)}개를 '{name}'(으)로 바꿨습니다. Ctrl+Z로 되돌립니다.")

    def _active_class(self, index):
        if not hasattr(self, "view"):
            return
        if self.task == "segment":
            self.view.class_id = self._active_class_id()
        self._selection_options()
        if self._ai_proposal is not None:
            self._show_proposal()

    def _active_class_id(self):
        value = self.class_combo.currentData()
        return value if type(value) is int else max(0, self.class_combo.currentIndex())

    def _change_class(self, index, class_id):
        if index not in self._indices() or not (
            0 <= class_id < len(self.names) or class_id == 255
        ):
            return
        indices = self.selected_indices if index in self.selected_indices else {index}
        self._reclass(indices, class_id)

    def _apply_class(self):
        self._reclass(self.selected_indices, self._active_class_id())

    def _reclass(self, indices, class_id):
        if self.task == "segment" and class_id == 0:
            # 분할의 0번은 배경이다. 배경으로 '클래스 변경'하면 객체 픽셀이 지워져 사라진다
            # (클래스 목록 맨 위 항목이라 첫 번째 클래스로 오해하기 쉽다). 지우기는 삭제로만 한다.
            self.status.setText("0번은 배경입니다. 선택 영역을 지우려면 '선택 삭제'를 쓰세요.")
            return
        indices = {
            i
            for i in indices
            if not self.object_meta.get(self._key(i), {}).get("locked")
        }
        if not indices:
            return
        self._checkpoint("클래스 변경")
        if self.task == "detect":
            for i in indices:
                self.rows[i]["class_id"] = class_id
        else:
            keys = {self._key(i) for i in indices if i >= 0}
            for i, shape in enumerate(self.document.shapes):
                if shape.get("operation") != "erase" and self.document.key(i) in keys:
                    shape["class_id"] = class_id
            self.document.invalidate()
            for i in indices:
                if i < -1:
                    pixels = self.document.render(copy_pixels=False)
                    mask = pixels == (-i - 2)
                    self.document.shapes.append(
                        self.document.raster_shape(
                            mask,
                            class_id,
                            object_id=uuid.uuid4().hex,
                            scope_class=-i - 2,
                        )
                    )
        self._changed()

    def _remove(self, index=None):
        indices = (
            self.selected_indices
            if index is None
            else (
                {index} if index not in self.selected_indices else self.selected_indices
            )
        )
        indices = {
            i
            for i in indices
            if i in self._indices()
            and not self.object_meta.get(self._key(i), {}).get("locked")
        }
        if not indices:
            return
        self._checkpoint("영역 삭제")
        if self.task == "detect":
            for i in sorted(indices, reverse=True):
                self.rows.pop(i)
                self.object_meta.pop(self.ids.pop(i), None)
        else:
            base = [i for i in indices if i < -1]
            keys = {self._key(i) for i in indices if i >= 0}
            self.document.shapes = [
                s
                for i, s in enumerate(self.document.shapes)
                if self.document.key(i) not in keys
                and s.get("scope_object") not in keys
            ]
            self.document.invalidate()
            for key in keys:
                self.object_meta.pop(key, None)
            for i in base:
                mask = self.document.render(copy_pixels=False) == (-i - 2)
                if mask.any():
                    self.document.shapes.append(
                        self.document.raster_shape(mask, 0, scope_class=-i - 2)
                    )
        self.selected = -1
        self.selected_indices = set()
        self.view.selected_set = set()
        self._changed()

    def _duplicate(self):
        indices = [
            i
            for i in sorted(self.selected_indices)
            if i in self._indices()
            and not self.object_meta.get(self._key(i), {}).get("locked")
        ]
        if not indices:
            return
        self._checkpoint("영역 복제")
        created = set()
        if self.task == "detect":
            for i in indices:
                row = copy.deepcopy(self.rows[i])
                x, y, w, h = row["coordinates"]
                row["coordinates"] = [
                    max(w / 2, min(1 - w / 2, x + 0.02)),
                    max(h / 2, min(1 - h / 2, y + 0.02)),
                    w,
                    h,
                ]
                self.rows.append(row)
                self.ids.append(uuid.uuid4().hex)
                created.add(len(self.rows) - 1)
        else:
            for i in indices:
                if i >= 0 and self.document.shapes[i]["kind"] == "polygon":
                    shape = copy.deepcopy(self.document.shapes[i])
                    p = shape["points"]
                    dx = min(0.02, 1 - max(p[::2]))
                    dy = min(0.02, 1 - max(p[1::2]))
                    shape["points"] = [
                        v + (dx if j % 2 == 0 else dy) for j, v in enumerate(p)
                    ]
                    shape["object_id"] = uuid.uuid4().hex
                else:
                    self.document._ensure_cache()
                    mask = self.document._owners == i
                    if not mask.any():
                        continue
                    shifted = np.zeros_like(mask)
                    shifted[1:, 1:] = mask[:-1, :-1]
                    if not shifted.any():
                        continue
                    shape = self.document.raster_shape(
                        shifted, self._class_id(i), object_id=uuid.uuid4().hex
                    )
                self.document.shapes.append(shape)
                created.add(len(self.document.shapes) - 1)
        self.selected_indices = created
        self.selected = next(iter(sorted(created)), -1)
        self._changed()

    def _split_components(self):
        if self.task != "segment" or len(self.selected_indices) != 1:
            self.status.setText("나눌 영역 한 개를 선택하세요.")
            return
        if self.pending or self.worker or self._grid_dialog:
            self.status.setText("진행 중인 그리기·저장·격자를 먼저 완료하세요.")
            return
        index = self.selected
        meta = copy.deepcopy(self.object_meta.get(self._key(index), {}))
        if meta.get("locked"):
            self.status.setText("잠긴 영역은 나누지 않습니다. 잠금을 먼저 해제하세요.")
            return
        try:
            shapes = self.document.component_shapes(index)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        if len(shapes) < 2:
            self.status.setText("선택 영역은 이미 하나로 연결되어 있습니다.")
            return
        from PySide6.QtWidgets import QDialogButtonBox, QLabel, QVBoxLayout

        revision = self.revision
        self._cancel_ai()
        preview = QDialog(self)
        preview.setWindowTitle("연결 부분 나누기 · 적용 전 미리보기")
        preview.resize(390, 170)
        layout = QVBoxLayout(preview)
        label = QLabel(
            f"선택 영역을 {len(shapes)}개의 연결 부분으로 나눕니다.\n"
            "노란 윤곽을 확인하세요. 클래스·마스크 픽셀·구멍은 유지합니다."
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).setText("분리 적용")
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            preview.accept
        )
        buttons.rejected.connect(preview.reject)
        layout.addWidget(buttons)
        self.view.grid_preview = []
        for shape in shapes:
            x, y, width, height = shape["bounds"]
            x1, x2 = (
                x / max(1, self.width_px - 1),
                (x + width - 1) / max(1, self.width_px - 1),
            )
            y1, y2 = (
                y / max(1, self.height_px - 1),
                (y + height - 1) / max(1, self.height_px - 1),
            )
            self.view.grid_preview.append({"points": [x1, y1, x2, y1, x2, y2, x1, y2]})
        self.view.viewport().update()
        try:
            accepted = preview.exec() == QDialog.DialogCode.Accepted
        finally:
            self.view.grid_preview = []
            self.view.viewport().update()
            preview.deleteLater()
        if not accepted:
            return
        if revision != self.revision or not self._source_image_current():
            self.status.setText("미리보기 중 원본이 수정됐습니다. 다시 선택하세요.")
            return
        before = self._capture()
        expected = self.document.render()
        undo_count = len(self.undo_stack)
        self._select_many({index})
        self._remove(index)
        created = set()
        for shape in shapes:
            key = uuid.uuid4().hex
            self.document.shapes.append({**shape, "object_id": key})
            self.document.objects[key] = copy.deepcopy(meta)
            created.add(len(self.document.shapes) - 1)
        self.document.invalidate()
        if not np.array_equal(expected, self.document.render(copy_pixels=False)):
            self._apply_state(before)
            self.undo_stack = self.undo_stack[:undo_count]
            self._changed()
            self.status.setText("마스크 픽셀을 보존할 수 없어 분리를 취소했습니다.")
            return
        self.undo_stack = self.undo_stack[:undo_count] + [("연결 부분 나누기", before)]
        self.selected_indices = created
        self.selected = min(created)
        self._changed()
        self.status.setText(
            f"{len(created)}개 영역으로 나눴습니다. 마스크 픽셀은 동일합니다."
        )

    def _set_flag(self, index, flag, value):
        if index not in self._indices():
            return
        key = self._key(index)
        if self.object_meta.get(key, {}).get(flag, False) == bool(value):
            return
        self._checkpoint("잠금·표시 변경")
        self.object_meta.setdefault(key, {})[flag] = bool(value)
        self._changed()

    def _render(self, region=None, *, reuse_cache=False):
        """reuse_cache: 창을 여는 첫 렌더링. 미리 읽기·검증에서 이미 렌더링한 문서의 캐시를 그대로 쓴다
        (4096²에서 약 0.2초). 편집 뒤의 전체 렌더링은 지금처럼 캐시를 버리고 다시 계산한다."""
        self._table_updating = True
        if self.task == "segment":
            self.rows = self.document.shapes
            self.object_meta = self.document.objects
            self.document.managed_cache = True
            if region is None and not reuse_cache:
                self.document.invalidate()
            self.view.set_document(self.document, self.names, self.selected, region)
        else:
            self.view.locked = {
                i
                for i, key in enumerate(self.ids)
                if self.object_meta.get(key, {}).get("locked")
            }
            self.view.hidden = {
                i
                for i, key in enumerate(self.ids)
                if self.object_meta.get(key, {}).get("hidden")
            }
            self.view.set_annotations(self.rows, self.names, self.selected)
        self._visible_indices = self._indices()
        self.selected_indices.intersection_update(self._visible_indices)
        self.view.selected_set = set(self.selected_indices)
        self.table.setRowCount(len(self._visible_indices))
        for row, index in enumerate(self._visible_indices):
            key = self._key(index)
            meta = self.object_meta.get(key, {})
            label = (
                f"R{meta['row']:02d} C{meta['column']:02d}"
                if "row" in meta and "column" in meta
                else str(row + 1)
            )
            if index < -1:
                label = "PNG " + str(-index - 2)
            item = self.table.item(row, 0)
            if item is None:
                item = QTableWidgetItem()
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, 0, item)
            item.setText(label)
            combo = self.table.cellWidget(row, 1)
            if combo is None:
                combo = NoWheelComboBox()
                self.table.setCellWidget(row, 1, combo)
                combo.currentIndexChanged.connect(self._table_class_changed)
            with QSignalBlocker(combo):
                expected = (
                    [*self.names, "학습 제외 (255)"]
                    if self.task == "segment"
                    else self.names
                )
                if [combo.itemText(i) for i in range(combo.count())] != expected:
                    combo.clear()
                    for cid, name in enumerate(self.names):
                        combo.addItem(name, cid)
                    if self.task == "segment":
                        combo.addItem("학습 제외 (255)", 255)
                combo.setProperty("index", index)
                combo.setCurrentIndex(combo.findData(self._class_id(index)))
                combo.setToolTip(combo.currentText())
                combo.setEnabled(not meta.get("locked", False))
            remove = self.table.cellWidget(row, 2)
            if remove is None:
                remove = QPushButton("×")
                self.table.setCellWidget(row, 2, remove)
                remove.clicked.connect(self._table_remove_clicked)
            remove.setProperty("index", index)
            remove.setEnabled(not meta.get("locked", False))
            for col, flag, invert in ((3, "locked", False), (4, "hidden", True)):
                check = self.table.cellWidget(row, col)
                if check is None:
                    check = QCheckBox()
                    self.table.setCellWidget(row, col, check)
                    check.setProperty("flag", flag)
                    check.setProperty("invert", invert)
                    check.toggled.connect(self._table_flag_changed)
                with QSignalBlocker(check):
                    check.setProperty("index", index)
                    check.setChecked(
                        not meta.get(flag, False) if invert else meta.get(flag, False)
                    )
        self._table_updating = False
        self._sync_table_selection()
        if self.selected in self._visible_indices:
            self.table.scrollToItem(
                self.table.item(self._visible_indices.index(self.selected), 0)
            )
        self._selection_options()
        self.object_summary.setText(
            f"영역 {len(self._visible_indices)}개 · 선택 {len(self.selected_indices)}개"
        )
        with QSignalBlocker(self.class_list):
            self.class_list.clear()
            for i, name in enumerate(self.names):
                # 분할의 0번은 클래스가 아니라 배경 (선택 영역에 적용하지 않는다)
                suffix = "  · 배경" if self.task == "segment" and i == 0 else ""
                self.class_list.addItem(f"{i}  {name}{suffix}")
        self._filter_classes(self.class_search.text())
        self.history.clear()
        for label, _ in reversed(self.undo_stack):
            self.history.addItem(label)
        self._buttons()
        if self.dirty:
            self.save_state.setText("편집 중 · 정답 저장 필요")

    def _buttons(self):
        busy = (
            self.worker is not None or getattr(self, "_save_queued", None) is not None
        )
        self.save_button.setEnabled(
            not busy
            and not self.pending
            and bool(self.names)
            and (self.dirty or self.save_empty)
        )
        self.review_button.setEnabled(
            not busy and not self.pending and bool(self.names)
        )
        self.finish_button.setEnabled(
            self.task == "segment" and len(getattr(self.view, "points", [])) >= 6
        )
        self.undo_button.setEnabled(bool(self.undo_stack) or self.pending)
        self.redo_button.setEnabled(bool(self.redo_stack))
        self.apply_class_button.setEnabled(
            bool(self.selected_indices) and bool(self.names)
        )
        self.add_class_button.setEnabled(
            self._add_class_callback is not None and not busy
        )
        self.previous_button.setEnabled(
            bool(self.navigation and self.navigation[0] > 0) and not busy
        )
        self.next_button.setEnabled(
            bool(self.navigation and self.navigation[0] + 1 < self.navigation[1])
            and not busy
        )
        self.grid_button.setEnabled(bool(self.selected_indices) and bool(self.names))

    def _pending(self, value):
        self.pending = bool(value)
        self._buttons()
        if value:
            self._draft_timer.start()

    def _empty_changed(self, value):
        if value:
            self.dirty = True
            self._draft_timer.start()
        self._buttons()

    def _set_mode(self, mode):
        self.view.set_mode(mode)
        for key, button in self.mode_buttons.items():
            button.setChecked(key == mode)
        if mode.startswith("ai_"):
            self.inspector.setCurrentWidget(self.ai_panel)
        self.view.setFocus()
        self.status.setText(
            {
                "erase": "지우기 대상: " + self.erase_scope.currentText(),
                "polygon": "클릭으로 점 추가 · Enter 완료 · Esc 취소",
                "select": (
                    "Shift+클릭 다중 선택 · 변 더블클릭 점 추가 · Alt+Delete 점 삭제"
                    if self.task == "segment"
                    else "Shift+클릭 다중 선택 · 선택 박스 드래그 이동 · Delete 삭제"
                ),
            }.get(mode, "Space 화면 이동 · 휠 커서 기준 확대")
        )

    def _finish_polygon(self):
        if self.task == "segment":
            self.view.finish_polygon()

    def _new_region(self):
        self._select_many(set())
        self.view.brush_object = None
        self.status.setText("다음 브러시 획은 새 영역으로 만듭니다.")

    def _filter_classes(self, value):
        for i in range(self.class_list.count()):
            self.class_list.item(i).setHidden(
                value.casefold() not in self.names[i].casefold()
            )

    def _class_list_selected(self, row):
        if 0 <= row < len(self.names):
            self.class_combo.setCurrentIndex(row)
            self._class_picked()

    def _add_class(self):
        if self._add_class_callback is None or self.worker:
            return
        if self.task == "segment" and len(self.names) >= 255:
            self.status.setText(
                "분할 클래스는 배경을 포함해 255개까지입니다. 255번 픽셀은 학습 제외 영역입니다."
            )
            return
        name, accepted = QInputDialog.getText(
            self,
            "클래스 추가",
            "클래스 이름"
            + (" (첫 클래스는 배경)" if self.task == "segment" else "")
            + ":",
        )
        if not accepted or not name.strip():
            return
        try:
            self.names = list(self._add_class_callback(name.strip()))
            if self.task == "segment":
                self.document.classes = len(self.names)
            self.class_combo.clear()
            self.class_combo.addItems(self.names)
            self.class_combo.setCurrentText(name.strip())
            self._render()
            for i in range(len(self.names)):
                self.class_combo.setItemData(i, i)
            if self.task == "segment":
                self.class_combo.addItem("학습 제외 영역 (255)", 255)
            self._active_class(self.class_combo.currentIndex())
            if self.task == "segment":
                self._baseline_state["document"].classes = len(self.names)
            self._baseline = self._label_fingerprint(self._baseline_state)
            if self.revision_provider:
                self.label_revision = self.revision_provider(self.image_path)
            self.revision += 1
            self._draft_timer.start()
        except Exception as exc:  # noqa: BLE001 - class-provider errors must stay visible in the editor.
            self.status.setText(str(exc))

    def _fit(self):
        self.view.fitInView(self.view.image_rect, Qt.AspectRatioMode.KeepAspectRatio)
        self.fit_scale = self.view.transform().m11()
        self._show_zoom()

    def focus_region(self, bounds, key="", seed=None, area=None):
        """데이터셋 영역 목록에서 연 영역을 확대하고 그 객체를 선택한다 (원본 픽셀 좌표).

        아주 작은 영역(점 결함)도 주변 맥락이 보이도록 최소 48px 정사각형에 여백을 둔다.
        """
        x, y, w, h = bounds
        side = max(w, h, 48)
        pad = side * 0.6
        cx, cy = x + w / 2, y + h / 2
        rect = QRectF(cx - side / 2 - pad, cy - side / 2 - pad, side + 2 * pad, side + 2 * pad)
        self.view.fitInView(rect.intersected(self.view.image_rect), Qt.AspectRatioMode.KeepAspectRatio)
        self._show_zoom()
        if self.task == "detect":
            index = int(key.split(":")[1]) if key.startswith("row:") else -1
        elif seed is not None:
            index = self.document.owner_at(int(seed[0]), int(seed[1]))
        else:
            index = -1
        if index == -1 or index not in self._indices():
            return
        if (self.task == "segment" and area is not None
                and int(np.count_nonzero(self.document._owners == index)) != int(area)):
            # 원본 PNG 층(클래스당 객체 하나)이나 떨어진 조각이 있는 객체는 선택하면 다른 blob까지
            # 함께 잡힌다. 확대만 하고, 이 영역만 다루는 방법을 안내한다.
            self.status.setText("이 영역은 같은 클래스의 다른 영역과 한 객체입니다. 이 영역만 편집하려면 "
                                "객체를 선택하고 '선택 영역의 연결 부분 나누기'를 쓰세요.")
            return
        self._select_many({index})
        self._select(index)

    def _actual_size(self):
        ratio = self.view.viewport().devicePixelRatioF()
        self.view.setTransform(QTransform.fromScale(1 / ratio, 1 / ratio))
        self._show_zoom()

    def _show_zoom(self):
        value = (
            self.view.transform().m11() * self.view.viewport().devicePixelRatioF() * 100
        )
        self.zoom_label.setText(f"{value:.0f}% · 원본 기준")
        with QSignalBlocker(self.zoom):
            self.zoom.setValue(round(value))

    def _zoom_value(self, value):
        scale = value / (100 * self.view.viewport().devicePixelRatioF())
        self.view.setTransform(QTransform.fromScale(scale, scale))
        self._show_zoom()

    def _zoom(self, direction):
        current = self.view.transform().m11()
        ratio = self.view.viewport().devicePixelRatioF()
        target = max(
            0.05 / ratio,
            min(16.0 / ratio, current * (1.2 if direction > 0 else 1 / 1.2)),
        )
        self.view.scale(target / current, target / current)
        self._show_zoom()

    def _command(self, command):
        if command in self.mode_buttons:
            self._set_mode(command)
        elif command == "undo":
            self._undo()
        elif command == "redo":
            self._redo()
        elif command == "fit":
            self._fit()
        elif command == "actual_size":
            self._actual_size()
        elif command == "delete":
            self._remove()
        elif command == "duplicate":
            self._duplicate()
        elif command == "save":
            self._save()
        elif command == "review_next":
            self._review_next()
        elif command == "grid":
            self._grid()
        elif command == "accept_proposal":
            self._apply_proposal()
        elif command == "cancel_proposal":
            self._cancel_ai()
        elif command == "overlay":
            self.annotation_visible.setChecked(not self.annotation_visible.isChecked())
        elif command == "brush_grow":
            self.brush_size.setValue(self.brush_size.value() + 2)
        elif command == "brush_shrink":
            self.brush_size.setValue(self.brush_size.value() - 2)
        elif command.startswith("class:"):
            cid = int(command.split(":")[1])
            cid += 1 if self.task == "segment" else 0
            if cid < len(self.names):
                self.class_combo.setCurrentIndex(cid)
                self._class_picked()
        elif command in {"previous", "next"}:
            self._navigate(-1 if command == "previous" else 1)
        elif command.startswith("fill:"):
            _, x, y = command.split(":")
            self._fill(int(x), int(y))
        elif command.startswith("nudge:"):
            self._nudge(command)

    def _nudge(self, command):
        _, key, amount = command.split(":")
        key, amount = int(key), int(amount)
        dx = (
            -amount
            if key == Qt.Key.Key_Left
            else amount
            if key == Qt.Key.Key_Right
            else 0
        )
        dy = (
            -amount if key == Qt.Key.Key_Up else amount if key == Qt.Key.Key_Down else 0
        )
        changes = {}
        if self.task == "detect":
            indices = [
                i for i in self.selected_indices if i >= 0 and i not in self.view.locked
            ]
            if not indices:
                return
            rects = [self.rows[i]["coordinates"] for i in indices]
            shift_x = max(
                -min(x - w / 2 for x, y, w, h in rects),
                min(1 - max(x + w / 2 for x, y, w, h in rects), dx / self.width_px),
            )
            shift_y = max(
                -min(y - h / 2 for x, y, w, h in rects),
                min(1 - max(y + h / 2 for x, y, w, h in rects), dy / self.height_px),
            )
            for i in indices:
                x, y, w, h = self.rows[i]["coordinates"]
                changes[i] = [
                    x + shift_x,
                    y + shift_y,
                    w,
                    h,
                ]
        else:
            keys = {
                self._key(i)
                for i in self.selected_indices
                if i >= 0 and i not in self.view.locked
            }
            originals = {
                i: shape
                for i, shape in enumerate(self.document.shapes)
                if self._key(i) in keys or shape.get("scope_object") in keys
            }
            points = []
            for shape in originals.values():
                if "points" in shape:
                    points.extend(zip(shape["points"][::2], shape["points"][1::2]))
                else:
                    x, y, w, h = shape["bounds"]
                    points.extend(
                        [
                            (
                                x / max(1, self.width_px - 1),
                                y / max(1, self.height_px - 1),
                            ),
                            (
                                (x + w - 1) / max(1, self.width_px - 1),
                                (y + h - 1) / max(1, self.height_px - 1),
                            ),
                        ]
                    )
            if not points:
                return
            a = max(
                -min(p[0] for p in points),
                min(1 - max(p[0] for p in points), dx / max(1, self.width_px - 1)),
            )
            b = max(
                -min(p[1] for p in points),
                min(1 - max(p[1] for p in points), dy / max(1, self.height_px - 1)),
            )
            for i, original in originals.items():
                shape = copy.deepcopy(original)
                if "points" in shape:
                    shape["points"] = [
                        v + (a if n % 2 == 0 else b)
                        for n, v in enumerate(shape["points"])
                    ]
                else:
                    shape["bounds"][0] += round(a * max(1, self.width_px - 1))
                    shape["bounds"][1] += round(b * max(1, self.height_px - 1))
                changes[i] = shape

        self._edit_many(changes)

    def _fill(self, x, y):
        if self.task != "segment":
            return
        try:
            region = self.document.connected_region(x, y)
            self._cancel_ai()
            self._ai_proposal = (self.revision, region)
            self._proposal_source = "채우기"
            self._show_proposal()
        except ValueError as exc:
            self.status.setText(str(exc))

    def _grid(self):
        if not self.selected_indices:
            self.status.setText("기준이 될 칩 박스나 영역 한 개를 선택하세요.")
            return
        if self._grid_dialog:
            self._grid_dialog.raise_()
            return
        index = self.selected
        if self.task == "detect":
            seed = self.rows[index]["coordinates"]
        else:
            bounds = self.document.bounds(index)
            if not bounds:
                return
            x, y, w, h = bounds
            seed = [
                (x + (w - 1) / 2) / max(1, self.width_px - 1),
                (y + (h - 1) / 2) / max(1, self.height_px - 1),
                (w - 1) / max(1, self.width_px - 1),
                (h - 1) / max(1, self.height_px - 1),
            ]
        self._grid_dialog = GridDialog(self, seed)
        self._grid_dialog.finished.connect(self._grid_closed)
        self._grid_dialog.show()

    def _grid_closed(self):
        self.view.grid_preview = []
        self.view.viewport().update()
        if self._grid_dialog:
            self._grid_dialog.deleteLater()
            self._grid_dialog = None

    def _apply_grid(self, candidates):
        if not candidates:
            raise ValueError("적용할 후보가 없습니다.")
        if any(not row["valid"] for row in candidates):
            raise ValueError("이미지 밖 후보를 제외하거나 기준점을 조정하세요.")
        self._checkpoint("격자 생성")
        created = set()
        cid = self._active_class_id()
        for row in candidates:
            key = uuid.uuid4().hex
            if self.task == "detect":
                self.rows.append(
                    {"class_id": cid, "coordinates": list(row["coordinates"])}
                )
                self.ids.append(key)
                index = len(self.rows) - 1
            else:
                shape = self.document.validate_shape(
                    {
                        "kind": "polygon",
                        "class_id": cid,
                        "points": list(row["points"]),
                        "object_id": key,
                    }
                )
                self.document.shapes.append(shape)
                index = len(self.document.shapes) - 1
            self.object_meta[key] = {"row": row["row"], "column": row["column"]}
            created.add(index)
        self.selected_indices = created
        self.selected = next(iter(sorted(created)), -1)
        self._changed()

    def _add_point(self, x, y):
        if self.task != "detect" or not self.names:
            return
        self.points.extend([x / self.width_px, y / self.height_px])
        if len(self.points) == 4:
            x1, y1, x2, y2 = self.points
            self.points = []
            if abs(x2 - x1) * self.width_px >= 1 and abs(y2 - y1) * self.height_px >= 1:
                self._create_box(
                    [(x1 + x2) / 2, (y1 + y2) / 2, abs(x2 - x1), abs(y2 - y1)]
                )
            else:
                self.status.setText("박스는 가로·세로 1픽셀 이상이어야 합니다.")
        self.pending = bool(self.points)
        self._buttons()

    _draft_navigation = 0

    def _ai_prompt(self, prompt):
        if self.task != "segment" or self._closing:
            return
        if not self._source_image_current():
            self.ai_status.setText(
                "원본 이미지가 변경됐습니다. 초안을 보존하고 이미지를 다시 여세요."
            )
            return
        self._cancel_ai()
        if "box" in prompt:
            self._prompt_box = list(prompt["box"])
            self.view.prompt_points = []
        if "point" in prompt:
            self.view.prompt_points.append((list(prompt["point"]), prompt["label"]))
        if getattr(self, "_prompt_box", None) is not None:
            prompt = dict(prompt, box=self._prompt_box)
        self._ai_request += 1
        request = self._ai_request
        if self.ai_worker:
            self.ai_worker.requestInterruption()
            self._queued_prompt = prompt
            return
        self._start_ai(prompt, request)

    def _start_ai(self, prompt, request):
        from core.annotation_assist import LocalSamAssist

        engine_key = (self.ai_model.currentData(), self.ai_device.currentIndex())
        if (
            self._ai_engine is None
            or getattr(self, "_ai_engine_key", None) != engine_key
        ):
            self._ai_engine = LocalSamAssist(
                engine_key[0], device="auto" if engine_key[1] == 0 else "cpu"
            )
            self._ai_engine_key = engine_key
        engine = self._ai_engine
        image = self.image_path
        revision = self.revision
        points = copy.deepcopy(self.view.prompt_points)

        def predict():
            return engine.predict(image, points=points, box=prompt.get("box"))

        self.ai_worker = AnnotationWrite(predict, self)
        self.ai_worker.finished.connect(lambda: self._ai_finished(request, revision))
        self.ai_worker.start()
        self.ai_status.setText("로컬 SAM2 처리 중 · 수동 편집 가능")
        self.ai_apply.setEnabled(False)

    def _ai_finished(self, request, revision):
        worker, self.ai_worker = self.ai_worker, None
        current = (
            request == self._ai_request
            and revision == self.revision
            and not self._closing
            and self._source_image_current()
        )
        if current and not worker.isInterruptionRequested():
            if worker.error:
                self.ai_status.setText(worker.error)
            else:
                self._ai_proposal = (revision, np.asarray(worker.value, dtype=bool))
                self.ai_apply.setEnabled(True)
                self._proposal_source = "AI"
                self.view.proposal = self._ai_proposal[1]
                self.ai_status.setText("AI 제안 · 확인 후 적용 또는 취소")
                try:
                    self._show_proposal()
                except (ValueError, TypeError) as exc:
                    self._cancel_ai()
                    self.ai_status.setText("AI 제안 오류: " + str(exc))
        elif request == self._ai_request and not self._closing:
            self.ai_status.setText(
                "원본 이미지 또는 정답이 변경되어 AI 응답을 폐기했습니다."
            )
        worker.deleteLater()
        queued = getattr(self, "_queued_prompt", None)
        self._queued_prompt = None
        if queued is not None and not self._closing:
            self._start_ai(queued, self._ai_request)

    def _source_image_current(self):
        try:
            return self.drafts.image_revision(self.image_path) == self._image_revision
        except OSError:
            return False

    def _show_proposal(self):
        if self._ai_proposal is None:
            return
        mask = self._ai_proposal[1]
        if not mask.any():
            self._cancel_ai()
            self.ai_status.setText("제안 영역이 비어 있습니다. 입력점을 바꾸세요.")
            return
        self.annotation_visible.setChecked(True)
        shape = self.document.raster_shape(
            mask,
            self._active_class_id(),
            protected_objects=[
                key for key, meta in self.object_meta.items() if meta.get("locked")
            ],
        )
        pixels = self.document.render(shape)
        self._proposal_bounds = shape["bounds"]
        self.view._set_overlay_pixels(pixels, self._proposal_bounds)
        self.view.proposal = mask
        self.proposal_label.setText(
            f"{getattr(self, '_proposal_source', 'AI')} 미리보기 · 이미지의 {100 * mask.mean():.1f}% · 적용 전 정답은 바뀌지 않습니다."
        )
        self.proposal_bar.show()

    def _apply_proposal(self):
        if self._ai_proposal is None or self._ai_proposal[0] != self.revision:
            return
        mask = self._ai_proposal[1]
        try:
            shape = self.document.raster_shape(
                mask, self._active_class_id(), object_id=uuid.uuid4().hex
            )
        except ValueError as exc:
            self.ai_status.setText(str(exc))
            return
        self._ai_proposal = None
        self.view.proposal = None
        self._create(shape)
        self.ai_apply.setEnabled(False)
        self.ai_status.setText("제안을 정답 편집에 적용했습니다. 저장해 확정하세요.")
        self.proposal_bar.hide()
        self.view.prompt_points = []
        self._prompt_box = None

    def _cancel_ai(self):
        had_proposal = self._ai_proposal is not None
        self._ai_request += 1
        self._queued_prompt = None
        if self.ai_worker:
            self.ai_worker.requestInterruption()
            self.ai_status.setText(
                "AI 요청을 취소했습니다. 수동 편집을 계속할 수 있습니다."
            )
        self._ai_proposal = None
        if hasattr(self, "proposal_bar"):
            self.proposal_bar.hide()
        if hasattr(self, "ai_apply"):
            self.ai_apply.setEnabled(False)
        if self.task == "segment" and hasattr(self, "view") and self.view.document:
            self.view.proposal = None
            if had_proposal:
                self.view._set_overlay_pixels(
                    self.document.render(copy_pixels=False),
                    getattr(self, "_proposal_bounds", None),
                )

    def _reset_ai_points(self):
        self._cancel_ai()
        self.view.prompt_points = []
        self._prompt_box = None
        self.view.viewport().update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            if self._ai_proposal is not None:
                self._cancel_ai()
            elif self.view.gesture or getattr(self.view, "points", []):
                self.view.cancel_gesture()
            else:
                self._select_many(set())
            event.accept()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "inspector"):
            if self.width() < 1100 and not getattr(self, "_compact", False):
                self.inspector.hide()
                self._compact = True
            elif self.width() >= 1100 and getattr(self, "_compact", False):
                self.inspector.show()
                self._compact = False
        if hasattr(self, "thumbnails"):
            self.thumbnails.setMaximumHeight(48 if self.height() < 600 else 82)
            self.status.setMaximumHeight(36 if self.height() < 600 else 16777215)


class DetectionAnnotationDialog(AnnotationWorkspace):
    def __init__(
        self,
        image_path,
        class_names,
        rows,
        save,
        parent=None,
        *,
        task="detect",
        **kwargs,
    ):
        super().__init__(
            image_path, class_names, rows, save, parent, task="detect", **kwargs
        )
