import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.planning.cem_planner import main


if __name__ == "__main__":
    main()
