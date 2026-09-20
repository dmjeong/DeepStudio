"""콘솔이 표현하지 못하는 로그 문자 때문에 작업이 중단되지 않도록 한다."""

import sys


def configure_console_output():
    # CP949 콘솔 및 부모 프로세스의 디코더와 호환되도록 기존 인코딩은 유지한다.
    # 표현할 수 없는 문자만 이스케이프하며 UTF-8 파일 로그는 그대로 보존한다.
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
