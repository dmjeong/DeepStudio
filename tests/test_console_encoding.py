"""Windows CP949 스트림에서 실제 데이터 로딩 로그를 검증한다."""

import contextlib
import io

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


def test_utf8_log_preserves_original_text():
    raw = io.BytesIO()
    with io.TextIOWrapper(raw, encoding="utf-8") as stream:
        with contextlib.redirect_stdout(stream):
            configure_console_output()
            print("📂 데이터 로드")
        stream.flush()
        assert raw.getvalue().decode("utf-8") == "📂 데이터 로드\n"
