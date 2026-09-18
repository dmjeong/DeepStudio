"""Qt는 진행 이벤트만 수신하고 실제 학습은 독립 프로세스에서 수행한다."""
from core.desktop_jobs import DesktopJob
from core.signals import TrainingSignals


class QtTrainingWorker(DesktopJob):
    def __init__(self, project, parent=None):
        from webapp.storage import digest, project_view
        from core.project import ProjectManager
        self.project = project
        payload = {"project": project_view(project), "project_digest": digest(ProjectManager.get_active_filepath(project)),
                   "persist_project": False}
        super().__init__("train", payload, parent)
        self.signals = TrainingSignals()
        self.event.connect(self._event)
        self.failed.connect(self.signals.training_error.emit)
        self.completed.connect(self._completed)

    def _event(self, event, args):
        signal = getattr(self.signals, event, None)
        if signal is not None:
            # JSON uses null for unavailable losses. Keep them unavailable when
            # crossing Qt's float signals instead of converting them to zero.
            args = list(args)
            float_fields = {"epoch_finished": (1, 2), "best_epoch_updated": (1, 2),
                            "training_finished": (0,), "batch_finished": (3,),
                            "lr_updated": (1,), "early_stopped": (1,)}
            for index in float_fields.get(event, ()):
                if index < len(args) and args[index] is None:
                    args[index] = float("nan")
            signal.emit(*args)

    def _completed(self, job):
        from core.desktop_jobs import desktop_manager
        from webapp.storage import read_json, restore_project
        path = desktop_manager().directory(job["id"]) / "project_result.json"
        if path.is_file():
            self.project.__dict__.update(restore_project(read_json(path)).__dict__)


class TrainWorker(QtTrainingWorker):
    pass




class PatchCoreWorker(QtTrainingWorker):
    pass
