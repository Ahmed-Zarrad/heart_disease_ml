"""Convenience entry point: run the full training/evaluation pipeline.

Equivalent to ``python -m src.train``.  Provided so the project can also be run
as a plain script:

    python run_pipeline.py                 # CV -> tune best -> evaluate -> save
    python run_pipeline.py --compare-only  # only the CV comparison table
    python run_pipeline.py --model "Random Forest"
"""
from src.train import main

if __name__ == "__main__":
    main()
