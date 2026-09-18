# Deep Vision Studio 20.5

Desktop training state fixes:

- Refresh supported Best metrics before validating a newly selected engine. Switching from EfficientNet Val Loss to 이전 모델/Custom, or between segmentation engines, must not validate the previous engine's metric.
- Prepare project persistence, the run snapshot and the device before clearing displayed results or disabling Start. Storage failures leave the page available for retry.
- Clear the previous project's hidden checkpoint path, progress, status, ETA, log and Best message when loading another project.
- Disable training settings while a worker owns the run. Restore settings after completion, cancellation and worker construction/start failures. Stop remains available while running.

Validation:

- Local dependency-free harness executed the actual Python widget methods with substitute controls: the mode transitions, storage failure and stale project state failed before these changes and passed after them. This is a state-logic check, not Qt rendering or model training.
- Added real Qt regression tests for mode changes, project switching, preparation failures and worker lifecycle. Run `python -m pytest tests/test_training_ui_state.py -v` in the application's dependency environment.
- No claim of visual verification on the user's Windows display or resolution. The user's reported full UI problem and identical model metrics still require reproduction in that environment.
