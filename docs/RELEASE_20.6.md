# Deep Vision Studio 20.6

## Confirmed chart reset defect

The confusion-matrix chart cleared its cells but retained the previous color scale. Its Figure still had two axes after `clear()` when it should contain only the main axes. Repeated redraws also removed color-scale axes directly rather than using the Colorbar lifecycle.

The chart now owns and removes its Colorbar before redraw and reset. A local check executed the actual chart class with a Matplotlib Agg canvas and a substitute non-Qt base: the old code retained two axes after clear; the new code passed 20 alternating matrix redraw/reset cycles with one axes after clear and no old cells or annotations. This verifies chart logic, not a full Qt application window.

## Reproducible UI evidence

`python tools/training_ui_proof.py` instantiates the real TrainingWidget and application theme using temporary projects. It exercises all classification training modes, B0 and B1, three viewport sizes and display scales 1, 1.25 and 1.5. It captures PNGs plus JSON geometry and failures. It sends explicitly synthetic result events and verifies that the displayed matrices change and clear correctly. It does not train models or establish accuracy.

The existing Windows and Linux CI test jobs run this check and upload its output with test artifacts. Missing Qt, timeouts and callback exceptions return failure rather than being skipped. Separate Qt regression tests cover startup errors and worker cancellation.

## Validation status at preparation

- Previous 20.5 local actual-method state checks: 5 tests passed, including engine changes, startup storage errors, project reset and worker lifecycle. These are not Qt render checks and were not rerun in the 20.6 session.
- Local chart drawing/reset check: 20 cycles passed.
- Full Qt rendering: not executed locally; PySide6 installation failed with no matching distribution available through the current environment.
- Remote UI execution: the previous CI retry, run 34946820984, failed before runner allocation. This is not a UI test pass.
- No customer screenshots, project files or production images were requested or used for these checks.
