import os
from pathlib import Path

ROOT_DIR = Path(os.path.abspath(__file__)).parent.parent.parent

DATA_DIR = ROOT_DIR / "data"
ACTIVATIONS_DIR = DATA_DIR / "activations"


MODEL_NAMES_MAPPING = {
    "llama_3.1_8b_instruct": "Llama-3.1-8B",
    "mistral_0.3_7b_instruct": "Mistral-0.3-7B",
    "phi_3.5_mini_instruct": "Phi-3.5-Mini",
}

PROMPT_NAMES_MAPPING = {
    "short_few_shot_sep": "Few-Shot",
    "short_zero_shot": "Zero-Shot",
    "short_few_shot_trivia": "Few-Shot-Trivia",
}

DATASET_NAMES_MAPPING = {
    "nq_open": "NQ-Open",
    "trivia_qa": "Trivia-QA",
    "squad": "SQuAD",
}
