"""Frozen V3 mining: reads only the reviewed training-role candidate pool."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from v4_phase2.pipeline import mine

if __name__ == "__main__":
    torch.set_num_threads(2)
    mine()
