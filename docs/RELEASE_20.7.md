> 과거 릴리스 기록입니다. 폐기된 엔진 관련 항목은 현재 기능이 아닙니다. 현재 지원 범위는 README를 참고하세요.

# Deep Vision Studio 20.7

Bug fix release (+0.1), based on deepstudio3 20.6.

## Result integrity

- Preserve the worker's effective training configuration, runtime, model source and unknown metadata when the desktop saves a Run. Keep requested settings separately.
- Record the originating job ID and checkpoint SHA-256. Display result identity from recorded settings.
- Keep a failed current web job separate from previous Runs, including after refresh. Match completed jobs by job ID or an explicit output Run ID.
- Repaint final desktop results using restored settings and class order. Rendering failures cannot prevent persistence and control recovery. A failed save cannot show the success dialog.

## Training display

- Use actual epoch coordinates for loss, metrics and learning rate. Missing observations remain gaps; repeated epoch events update the same point.
- Preserve unavailable validation losses as N/A across JSON and Qt float signals.
- Announce the resume baseline before training and estimate remaining time from this session's completed epochs.
- Display batch progress for custom, EfficientNet and 폐기된 외부 엔진 workers. Keep progress below 100 until result persistence succeeds.
- Stack settings and monitoring in narrow desktop windows, use two-column metric cards, and make both panels scrollable.

## Verification and release

- Add lightweight result integrity tests that do not require Qt or model downloads. They do not replace Qt rendering or real training tests.
- Extend the official gradient comparison to EfficientNet B1.
- Require five-model training comparison, including independent prediction/metric checks, before tagging.
- Include deepstudio3 in verified tag and release conditions. A skipped or failed required check prevents a release tag.

## Scope and outstanding verification

This is the first correctness repair batch. It does not claim production accuracy or deployment performance improvements. The ONNX/C++ image preprocessing alignment, shared engine capability definitions and expanded layer debugging interface remain separate follow-up work.

Qt rendering and real pretrained training require the configured CI runtimes. Check the workflow results and artifacts for this exact commit before considering the release verified.
