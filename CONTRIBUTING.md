# Contributing

For a method change, describe its inputs, outputs, and relationship to the existing estimators. Add a focused test and update the relevant method and experiment guide.

For a result, specify the dataset, responder split, label access, metric, baseline, and uncertainty calculation. Keep different protocols separate and include negative results when they affect the conclusion. Update the report guide when adding or replacing a table.

Run `pytest -q` and `ruff check .` before submitting a change. Keep credentials, raw datasets, model caches, and checkpoints out of the repository.
