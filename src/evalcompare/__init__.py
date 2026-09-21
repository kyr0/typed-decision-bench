"""/ evalcompare: the metrics & comparison library behind scripts/metrics.py.

One direction of dependency between four modules: loader (stats JSONL ->
DataFrames) -> metrics (registry of names/directions/formats) -> analysis
(capability x run comparison vs a baseline) -> report (self-contained plotly
dashboard). scripts/score.py writes exactly the column names the registry
knows, so metric names mean the same thing everywhere.
"""
