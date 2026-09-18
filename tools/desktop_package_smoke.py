"""배포 EXE 내부에서 GUI, 모델 로드, 독립 추론과 영구 결과 복원을 검증."""
from pathlib import Path
import os
import tempfile
import time


def run(output):
    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import QTimer, Qt
    from PIL import Image
    import torch
    from efficientnet import EfficientNet
    from checkpoint import make_checkpoint_metadata
    from core.project import ProjectManager
    from core.desktop_jobs import desktop_manager, PersistentInferenceCache
    from webapp.storage import read_json, write_json
    from app.main_window import MainWindow
    from core.version import APP_VERSION
    def stage(name):
        print("SMOKE_STAGE:", name, flush=True)
    def failed_dialog(parent, title, message, *args, **kwargs):
        raise RuntimeError(f"{title}: {message}")
    QMessageBox.warning = failed_dialog
    QMessageBox.critical = failed_dialog
    stage("imports_ready")
    if not torch.version.cuda:
        raise RuntimeError("GPU 지원 배포에 CPU 전용 PyTorch 포함됨")
    library = Path(torch.__file__).parent / "lib"
    for name in ("torch_cuda.dll", "c10_cuda.dll"):
        if not (library / name).is_file():
            raise RuntimeError(f"배포 CUDA DLL 누락: {name}")
    stage("cuda_runtime_bundled")
    import torchvision
    kept = torchvision.ops.nms(torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]]),
                               torch.tensor([.9, .8]), .5)
    if kept.tolist() != [0]:
        raise RuntimeError("패키지의 torchvision NMS 연산 검증 실패")
    stage("native_nms_ready")
    torch.set_num_threads(1)
    app = QApplication.instance() or QApplication([])
    os.environ["DEEP_STUDIO_DESKTOP_STATE_DIR"] = str(Path(output).parent / "smoke-state")
    window = MainWindow()
    stage("window_ready")
    ticks = []
    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start()
    def wait(predicate, limit=150):
        deadline = time.monotonic() + limit
        while predicate():
            app.processEvents()
            if time.monotonic() > deadline:
                raise TimeoutError("패키지 작업 완료 시간 초과")
            time.sleep(.01)
        app.processEvents()
    with tempfile.TemporaryDirectory(prefix="studio-smoke-") as folder:
        root = Path(folder)
        project = ProjectManager.create_new("검증 프로젝트", "classify", str(root / "project"), ["OK", "NG"])
        ProjectManager.save(project)
        window.set_project(project)
        window.show()
        wait(lambda: getattr(window.dataset_page, "_scan_worker", None) is not None)
        stage("dataset_ready")
        weights = root / "fixture.pt"
        model = EfficientNet(num_classes=2, in_channels=1).eval()
        checkpoint = make_checkpoint_metadata("classify", 2, ["OK", "NG"], (64, 64), 1)
        checkpoint.update(engine="efficientnet", model_config=model.checkpoint_config(), model_state_dict=model.state_dict())
        torch.save(checkpoint, weights)
        image_path = root / "image.png"
        Image.new("RGB", (96, 64), (170, 80, 20)).save(image_path)
        page = window.inference_page
        window.content_stack.setCurrentWidget(page)
        page.ckpt_edit.setText(str(weights))
        before = len(ticks)
        stage("model_inspection_start")
        page._start_model_inspection()
        wait(lambda: page._model_inspection is not None)
        if not page._process_model:
            raise RuntimeError(page.model_info.text())
        stage("model_inspection_ready")
        page._batch_images = [str(image_path)]
        page._current_image = str(image_path)
        page.gradcam_checkbox.setChecked(True)
        stage("inference_start")
        page._run_inference()
        wait(lambda: page._inference_worker is not None)
        stage("inference_ready")
        result = page._inference_results[str(image_path)]
        if result.status == "error" or result.inference_sec is None:
            raise RuntimeError(str(result))
        if len(ticks) - before < 3:
            raise RuntimeError("GUI 이벤트 루프 응답 없음")
        job = next(j for j in desktop_manager().list(limit=100) if j["kind"] == "infer")
        saved = read_json(desktop_manager().directory(job["id"]) / "results/0.json")
        restored = PersistentInferenceCache(desktop_manager().directory(job["id"]))
        if restored.get(str(image_path)) is None or not saved["cache_ready"]:
            raise RuntimeError("영구 결과 복원 실패")
        page._restore_inference_job(job["id"])
        stage("defect_preview_start")
        normal = Path(project.data.train_dir) / "OK" / "normal.png"
        Image.new("RGB", (96, 64), (128, 128, 128)).save(normal)
        defect = window.defect_gen_page
        window.content_stack.setCurrentWidget(defect)
        defect._start_generation(preview=True)
        wait(lambda: defect._worker is not None)
        if not defect._job_result or defect._job_result["status"] != "completed" or defect.result_list.count() != 1:
            raise RuntimeError("배포 EXE 합성 후보 생성 실패: " + defect.progress_label.text())
        defect.result_list.item(0).setCheckState(Qt.CheckState.Checked)
        defect._publish()
        wait(lambda: defect._worker is not None)
        if defect._job_result["status"] != "completed" or defect._job_result["output"].get("saved") != 1:
            raise RuntimeError("배포 EXE 합성 검수 저장 실패: " + defect.progress_label.text())
        stage("defect_review_saved")
        screenshot = Path(output).with_suffix(".png")
        window.grab().save(str(screenshot))
        write_json(output, {"version": APP_VERSION, "frozen": bool(getattr(__import__('sys'), 'frozen', False)),
            "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gui_ticks": len(ticks) - before, "inference_status": result.status,
            "gradcam_status": result.gradcam_status, "persistent_results": True, "defect_review_saved": True})
        desktop_manager().close()
        window.project = None  # 종료 확인 대화상자는 자동 배포 검증에서 생략
        window.close()
    timer.stop()
