"""Qt 없이 실행하며 진행 이벤트와 협력적 중단을 받는 학습 기반 클래스."""

TRAINING_EVENTS = (
    "epoch_finished", "best_epoch_updated", "batch_finished", "training_finished",
    "training_error", "log_message", "progress_updated", "early_stopped",
    "lr_updated", "eval_finished", "layer_debug",
)


class CallbackSignal:
    def __init__(self, callback=None):
        self.callbacks = [callback] if callback else []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, *args):
        for callback in tuple(self.callbacks):
            callback(*args)


class TrainingEvents:
    def __init__(self, callback=None):
        for name in TRAINING_EVENTS:
            emit = (lambda *args, event=name: callback(event, args)) if callback else None
            setattr(self, name, CallbackSignal(emit))


class TrainingEngine:
    def __init__(self, project, *, signals=None, should_stop=None):
        self.project = project
        self.signals = signals if signals is not None else TrainingEvents()
        self._cancel_check = should_stop or (lambda: False)
        self._cancel_requested = False

    @property
    def _stop_requested(self):
        return self._cancel_requested or self._cancel_check()

    @_stop_requested.setter
    def _stop_requested(self, value):
        self._cancel_requested = bool(value)
