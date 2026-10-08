"""
Execution Entrypoint: Gemma 2 2B Math Distractor Generation & Verification Pipeline.

Processes all 250 intersection MATH items:
1. Loads Gemma 2 2B (in 4-bit) from the HDD cache on drive D:.
2. Queries Gemma for educational student misconceptions / calculation slips.
3. Strictly verifies each generated distractor (FA != GT, clean number).
4. Saves intermediate checkpoints every 5 items.
5. Backs up and updates final_filtered_evaluation_set_500.json with high-quality distractors.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data.generate_math_distractors import run_gemma_distractor_generation

if __name__ == "__main__":
    run_gemma_distractor_generation()
