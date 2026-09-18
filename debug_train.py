"""VS Code에서 실제 EfficientNet 학습 엔진을 직접 디버깅한다."""

import argparse
from itertools import islice
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "gui")]


class LimitedLoader:
    def __init__(self, loader, batches):
        self.loader, self.batches = loader, batches

    def __iter__(self):
        return islice(iter(self.loader), self.batches)

    def __len__(self):
        return min(len(self.loader), self.batches)

    def __getattr__(self, name):
        return getattr(self.loader, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="기존 .dvproj 파일")
    parser.add_argument("--variant", choices=("efficientnet_b0", "efficientnet_b1"), default="efficientnet_b0")
    parser.add_argument("--weights", help="선택: 추가 학습할 로컬 가중치")
    parser.add_argument("--device", default="auto", help="auto는 GPU 우선, cuda는 GPU 필수")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", nargs="+", default=["features.0", "features.1.*", "features.8", "classifier.1"])
    parser.add_argument("--keep-values", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batches < 1:
        parser.error("epochs와 batches는 양수 필요")
    import numpy as np
    import torch
    from core.project import ProjectManager
    from core.efficientnet_trainer import EfficientNetTrainWorker
    from core.training_engine import TrainingEvents
    from layer_debug import LayerInspector
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    project = ProjectManager.load(args.project)
    if project.task != "classify":
        parser.error("분류 프로젝트 선택 필요")
    project.project_dir = str(Path(project.project_dir) / "debug-runs")
    project.training.training_mode = "efficientnet_transfer" if args.weights else "efficientnet_finetune"
    project.training.efficientnet_model = args.variant
    project.training.device = args.device
    project.training.epochs = args.epochs
    project.training.use_amp = False
    project.model.pretrained_weights = args.weights or ""
    def event(name, values):
        if name in {"log_message", "training_error"}:
            print(*values, flush=True)
    trainer = EfficientNetTrainWorker(project, signals=TrainingEvents(event))
    trainer.debug_num_workers = 0
    trainer.debug_fp32 = True
    trainer.debug_batch_limit = args.batches
    inspector = LayerInspector(args.layers, keep_values=args.keep_values)
    trainer.debug_observer = inspector.attach
    create_loaders = trainer._create_dataloaders
    def limited_loaders(*arguments, **keywords):
        loaders = create_loaders(*arguments, **keywords)
        return tuple(LimitedLoader(loader, args.batches) for loader in loaders)
    trainer._create_dataloaders = limited_loaders
    try:
        # UI와 같은 학습 루프. 예외는 디버거에 그대로 전달한다.
        trainer._run_training()
        output = Path(project.project_dir) / "layer-trace.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(inspector.records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"레이어 관찰 기록: {output}")
    finally:
        inspector.close()


if __name__ == "__main__":
    main()
