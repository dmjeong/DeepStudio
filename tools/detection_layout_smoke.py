"""Render the detection editor with the application theme at desktop DPI scales."""
import argparse
import base64
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def check(scale):
    from PIL import Image, ImageDraw
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    sys.path.insert(0, str(ROOT / 'gui'))
    from widgets.obb_annotation import OBBAnnotationDialog
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet((ROOT / 'gui/resources/styles/dark_theme.qss').read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'inspection.png'
        fixture = Image.new('RGB', (1000, 700), '#353b44')
        draw = ImageDraw.Draw(fixture)
        for y in range(60, 650, 140):
            for x in range(50, 950, 190):
                draw.rounded_rectangle((x,y,x+145,y+100), radius=10, fill='#747d88', outline='#9ba5b0', width=3)
        draw.line((285, 247, 345, 260), fill='#e4e7eb', width=4)
        fixture.save(path)
        dialog = OBBAnnotationDialog(str(path), ['surface_mark', 'edge_chip'], [], lambda rows: None, task='detect', add_class=lambda name: ['surface_mark','edge_chip',name])
        dialog.show()
        for width, height in [(1080,800), (900,680)]:
            dialog.resize(width, height)
            app.processEvents()
            dialog._fit()
            assert dialog.width() == width, ('editor forced wider', scale, width, dialog.width())
            assert dialog.height() == height, ('editor forced taller', scale, height, dialog.height())
            controls = list(dialog.mode_buttons.values()) + [dialog.undo_button, dialog.redo_button]
            for i, button in enumerate(controls):
                assert dialog.rect().contains(button.geometry())
                for other in controls[i+1:]:
                    assert not button.geometry().intersects(other.geometry()), 'overlapping tools'
            assert dialog.view.viewport().width() > 350
            assert dialog.view.viewport().height() >= 250
        start = dialog.view.mapFromScene(QPointF(265,225))
        end = dialog.view.mapFromScene(QPointF(365,275))
        QTest.mousePress(dialog.view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(dialog.view.viewport(), end)
        QTest.mouseRelease(dialog.view.viewport(), Qt.MouseButton.LeftButton, pos=end)
        app.processEvents()
        assert len(dialog.rows) == 1, 'drag did not create a box'
        assert dialog.selected == 0
        assert dialog.table.rowCount() == 1
        output = ROOT / 'detection-layout'
        output.mkdir(exist_ok=True)
        png = output / f'detection-{scale}.png'
        assert dialog.grab().save(str(png))
        if scale == '1':
            print('DETECTION_EDITOR_PNG=' + base64.b64encode(png.read_bytes()).decode('ascii'))
        print('DETECTION_LAYOUT_OK=' + scale)
        dialog.dirty = False
        dialog.close()
        app.processEvents()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale')
    args = parser.parse_args()
    if args.scale:
        check(args.scale)
    else:
        for value in ('1', '1.25', '1.5'):
            env = dict(os.environ, QT_QPA_PLATFORM='offscreen', QT_SCALE_FACTOR=value)
            subprocess.run([sys.executable, __file__, '--scale', value], env=env, check=True, timeout=60)
