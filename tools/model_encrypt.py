"""Encrypt any existing Studio ONNX export. Run without arguments for dialogs."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from model_crypto import create_key, encrypt_config, read_key


def main():
    parser = argparse.ArgumentParser(description="Studio 모델·설정 암호화")
    parser.add_argument("--new-key", help="새 32바이트 암호키 파일 (기존 파일은 덮어쓰지 않음)")
    parser.add_argument("--config", help="기존 export의 JSON (SAM2는 sam2.json)")
    parser.add_argument("--key", help="기존 32바이트 암호키")
    parser.add_argument("--output", help="출력 .dvsenc")
    args = parser.parse_args()
    if len(sys.argv) == 1:
        from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
        app = QApplication.instance() or QApplication([])
        app.setApplicationName("Deep Vision Studio Model Encryption")
        try:
            config, _ = QFileDialog.getOpenFileName(None, "내보낸 모델의 JSON 선택", "", "JSON (*.json)")
            if not config:
                return
            key_path, _ = QFileDialog.getSaveFileName(None, "새 암호키를 별도로 보관할 위치", "", "Key (*.key)")
            if not key_path:
                return
            output, _ = QFileDialog.getSaveFileName(None, "암호화 모델 저장", "", "Encrypted model (*.dvsenc)")
            if not output:
                return
            if Path(output).resolve() == Path(key_path).resolve():
                raise ValueError("암호키와 모델 저장 위치가 같을 수 없습니다.")
            create_key(key_path)
            encrypt_config(config, output, read_key(key_path))
            QMessageBox.information(None, "완료", "암호화와 복원 검증을 완료했습니다. 키를 모델과 별도로 보관하세요.")
        except Exception as exc:
            QMessageBox.critical(None, "실패", str(exc))
        return
    if args.new_key:
        create_key(args.new_key)
    if args.config:
        if not args.output or not (args.key or args.new_key):
            parser.error("--config 사용 시 --output 및 --key 또는 --new-key가 필요합니다")
        if Path(args.output).resolve() == Path(args.key or args.new_key).resolve():
            parser.error("암호키와 모델 저장 위치가 같을 수 없습니다")
        encrypt_config(args.config, args.output, read_key(args.key or args.new_key))
        print("암호화 및 복원 일치 검증 완료")
    elif not args.new_key:
        parser.error("--new-key 또는 --config를 지정하세요")


if __name__ == "__main__":
    main()
