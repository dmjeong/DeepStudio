"""학습 경과 시간: 벽시계 변경에 영향받지 않는 초와 누적 시:분:초."""

import math
import time


def format_hms(seconds):
    if seconds is None or not math.isfinite(float(seconds)) or seconds < 0:
        return "N/A"
    whole = int(seconds)
    hours, remainder = divmod(whole, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainingClock:
    def __init__(self, clock=None):
        self.clock = clock or time.perf_counter
        self.started = self.clock()
        self.epoch_started = None

    def start_epoch(self):
        self.epoch_started = self.clock()

    def end_epoch(self):
        now = self.clock()
        duration = None if self.epoch_started is None else max(0., now - self.epoch_started)
        self.epoch_started = None
        return duration, max(0., now - self.started)

    def elapsed(self):
        return max(0., self.clock() - self.started)


def record_epoch_time(clock, history, log, epoch):
    duration, elapsed = clock.end_epoch()
    history.setdefault("epoch_time_sec", []).append(duration)
    history.setdefault("elapsed_time_sec", []).append(elapsed)
    log(f"  Epoch {epoch} 시간 | 소요 {format_hms(duration)} | 누적 {format_hms(elapsed)}")


def log_total_time(log, seconds):
    log(f"  총 학습시간: {format_hms(seconds)} ({seconds:.3f}초)")
