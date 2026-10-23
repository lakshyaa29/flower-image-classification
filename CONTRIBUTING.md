# Contributing

Use a new branch for changes. Keep image credits and reviewer attribution intact, and document any changes to dataset selection, preprocessing, or experiment budgets.

Install development dependencies, run `python tools/check_repository.py`, and run `python -m unittest discover -s tests -v`. Fixture tests should remain small and should not download data or iterate the real test set. If helpers change, rebuild clean notebook templates with the five `tools/build*notebook.py` scripts; preserve completed notebook snapshots.

The recorded test set has already been evaluated. New tuning informed by its results requires a newly reserved evaluation protocol. Clearly distinguish new experiments from the accepted final records and from bounded development checks.
