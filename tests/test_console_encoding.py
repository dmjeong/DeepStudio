"""Windows CP949 스트림에서 실제 데이터 로딩 로그를 검증한다."""

import contextlib
import io
import os

import pytest

from core.console_encoding import configure_console_output
from core.training_engine import TrainingEngine
from tests.test_training_contracts import dataset_namespace, write_image


def test_training_initialization_prevents_cp949_logging_crash(tmp_path):
    pytest.importorskip("cv2")
    namespace = dataset_namespace()
    write_image(tmp_path / "정상📂" / "이미지.png")
    raw = io.BytesIO()
    with io.TextIOWrapper(raw, encoding="cp949", errors="strict") as stream:
        with pytest.raises(UnicodeEncodeError):
            stream.write("📂 Classification 데이터셋 로드")
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            TrainingEngine(object())
            dataset = namespace["ClassificationDataset"](str(tmp_path))
            print("📂 학습 시작")
            assert len(dataset) == 1
            assert stream.encoding == "cp949"
        stream.flush()
        output = raw.getvalue().decode("cp949")
        assert "클래스 수: 1" in output
        assert "\\U0001f4c2" in output


@pytest.mark.parametrize("newline,ending", [(None, os.linesep), ("\n", "\n"), ("\r\n", "\r\n")])
def test_utf8_log_preserves_original_text(newline, ending):
    raw = io.BytesIO()
    with io.TextIOWrapper(raw, encoding="utf-8", newline=newline) as stream:
        with contextlib.redirect_stdout(stream):
            configure_console_output()
            print("📂 데이터 로드")
        stream.flush()
        assert raw.getvalue().decode("utf-8") == "📂 데이터 로드" + ending


@pytest.mark.parametrize("encoding", ["cp1252", "cp949"])
def test_real_export_completes_with_legacy_windows_console(tmp_path, encoding):
    from example.generate_export_test_assets import generate
    destination = tmp_path / "모델📂"
    raw = io.BytesIO()
    with io.TextIOWrapper(raw, encoding=encoding, errors="strict") as stream:
        with contextlib.redirect_stdout(stream):
            generate(destination)
        assert stream.errors == "strict"  # The caller's stream settings are not changed.
        stream.flush()
        output = raw.getvalue().decode(encoding)
        assert "\\u" in output or "\\U" in output
    assert (destination / "libreyolo.onnx").is_file()
    assert (destination / "libreyolo.json").is_file()
    assert (destination / "expected.json").is_file()
    assert not list(destination.glob("*.export-error.json"))


def test_export_preserves_custom_logger_errors(tmp_path):
    from export_onnx import export_checkpoint

    def callback(message):
        message.encode("ascii")

    with pytest.raises(UnicodeEncodeError):
        export_checkpoint(tmp_path / "unused.pt", tmp_path / "unused.onnx", log=callback)
