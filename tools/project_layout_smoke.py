"""Render the real dark project page and check layout at Windows display scales."""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def check(scale):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QLabel, QGroupBox, QPushButton
    sys.path.insert(0, str(ROOT / "gui"))
    from widgets.project_widget import ProjectWidget

    app = QApplication.instance() or QApplication([])
    app.setStyleSheet((ROOT / "gui/resources/styles/dark_theme.qss").read_text(encoding="utf-8"))
    with patch("widgets.project_widget.ProjectManager.list_recent", return_value=list(map(str, range(5)))), \
            patch("widgets.project_widget.ProjectManager.load", return_value=SimpleNamespace(name="Inspection", task="obb")):
        page = ProjectWidget()
    try:
        page.show()
        columns_seen = set()
        sizes = ((1280, 800), (1080, 660), (880, 600), (720, 560), (440, 480), (1080, 660))
        for width, height in sizes:
            page.resize(width, height)
            page.scroll.verticalScrollBar().setValue(0)
            QTest.qWait(100)
            app.processEvents()
            assert page.width() == width, ("page forced wider", width, page.width())
            assert page.content.width() <= page.scroll.viewport().width(), (
                "content clipped horizontally", width, page.content.width(), page.scroll.viewport().width(),
                [(section.objectName(), section.minimumSizeHint().width()) for section in page.content.findChildren(QGroupBox)])
            cards = list(page.task_cards.values())
            columns_seen.add(page._task_columns)
            for index, card in enumerate(cards):
                rect = card.geometry()
                assert page.task_group.contentsRect().contains(rect), ("card outside group", card.task_key, rect)
                for other in cards[index + 1:]:
                    assert not rect.intersects(other.geometry()), ("cards overlap", card.task_key, other.task_key)
                labels = card.findChildren(QLabel)
                for label_index, label in enumerate(labels):
                    label_rect = label.geometry()
                    assert card.rect().contains(label_rect), ("label outside card", card.task_key, label.text())
                    for other in labels[label_index + 1:]:
                        assert not label_rect.intersects(other.geometry()), ("labels overlap", label.text(), other.text())
                    if label.wordWrap():
                        required = label.fontMetrics().boundingRect(
                            label.contentsRect(), int(Qt.TextFlag.TextWordWrap), label.text()).height()
                        assert label.contentsRect().height() >= required, ("text clipped", card.task_key, label.text(), required, label.height())
                color = card.grab().toImage().pixelColor(6, card.height() // 2)
                assert max(color.red(), color.green(), color.blue()) < 110, ("card background is not dark", color.name())
            sections = page.content.findChildren(QGroupBox, options=Qt.FindChildOption.FindDirectChildrenOnly)
            section_rects = [section.geometry() for section in sections] + [page.create_btn.geometry()]
            for index, rect in enumerate(section_rects):
                for other in section_rects[index + 1:]:
                    assert not rect.intersects(other), ("project sections overlap", rect, other)
            assert page.scroll.verticalScrollBar().maximum() > 0, "long content needs a scrollbar"
            page.scroll.ensureWidgetVisible(page.create_btn)
            QTest.qWait(20)
            pos = page.create_btn.mapTo(page.scroll.viewport(), QPoint(0, 0))
            assert 0 <= pos.y() and pos.y() + page.create_btn.height() <= page.scroll.viewport().height(), "create button inaccessible"
            last = page.recent_layout.itemAt(page.recent_layout.count() - 1).widget()
            assert isinstance(last, QPushButton)
            page.scroll.ensureWidgetVisible(last)
            QTest.qWait(20)
            assert page.scroll.viewport().rect().contains(last.mapTo(page.scroll.viewport(), last.rect().center())), "recent project inaccessible"
        assert columns_seen == {1, 2, 3}, ("responsive columns", columns_seen)
        card = page.task_cards["obb"]
        card.setFocus()
        QTest.keyClick(card, Qt.Key.Key_Space)
        assert page._selected_task == "obb"
        assert sum(card._selected for card in cards) == 1
        page.scroll.verticalScrollBar().setValue(0)
        QTest.qWait(40)
        directory = ROOT / "project-layout"
        directory.mkdir(exist_ok=True)
        snapshot = directory / f"project-{scale}.png"
        assert page.grab().save(str(snapshot))
        print("PROJECT_LAYOUT_OK " + json.dumps({"scale": scale, "sizes": len(sizes), "columns": sorted(columns_seen)}), flush=True)
        if scale == "1":
            print("PROJECT_LAYOUT_PNG=" + base64.b64encode(snapshot.read_bytes()).decode("ascii"), flush=True)
    finally:
        page.close()
        page.deleteLater()
        app.processEvents()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale")
    args = parser.parse_args()
    if args.scale:
        check(args.scale)
    else:
        for scale in ("1", "1.25", "1.5"):
            env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "QT_SCALE_FACTOR": scale}
            subprocess.run([sys.executable, __file__, "--scale", scale], env=env, check=True, timeout=60)
