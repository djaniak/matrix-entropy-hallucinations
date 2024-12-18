"""Implement semantic entropy metrics and entailment models.

This code is adapted from:
https://github.com/jlko/semantic_uncertainty

Which contains code to reproduce experiments from the Nature submission
'Detecting Hallucinations in Large Language Models Using Semantic Entropy'.

The original codebase this builds on is:
https://github.com/lorenzkuhn/semantic_uncertainty
"""

import logging
import os
from typing import Any, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BaseEntailment:
    """Base class for entailment models."""

    def check_implication(self, text1: str, text2: str, *args: Any, **kwargs: Any) -> int:
        """Check if text1 entails text2."""
        raise NotImplementedError


class EntailmentDeberta(BaseEntailment):
    """DeBERTa model for natural language inference."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli"
        ).to(device)

    def check_implication(self, text1: str, text2: str, *args: Any, **kwargs: Any) -> int:
        """Check if text1 entails text2.

        Returns:
            0: contradiction
            1: neutral
            2: entailment
        """
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        prediction = int(torch.argmax(F.softmax(logits, dim=1)).cpu().item())

        if os.environ.get("DEBERTA_FULL_LOG", False):
            logging.info("Deberta Input: %s -> %s", text1, text2)
            logging.info("Deberta Prediction: %s", prediction)

        return prediction


class EntailmentLLM(BaseEntailment):
    """Base class for LLM-based entailment."""

    def equivalence_prompt(self, text1: str, text2: str, question: str) -> str:
        raise NotImplementedError

    def predict(self, prompt: str, temperature: float) -> str:
        raise NotImplementedError

    def check_implication(self, text1: str, text2: str, example: Optional[dict] = None) -> int:
        if example is None:
            raise ValueError("Example must be provided")

        prompt = self.equivalence_prompt(text1, text2, example["question"])
        response = self.predict(prompt, temperature=0.02).lower()
        print("response", response)

        if "entailment" in response:
            return 2
        elif "contradiction" in response:
            return 0
        else:
            if "neutral" not in response:
                logging.warning("Unexpected response format - defaulting to neutral")
            return 1


class EntailmentLlama(EntailmentLLM):
    """Llama-based entailment model."""

    def __init__(self, llm: Any, generation_config: Any, device: str) -> None:
        self.model = llm.llm
        self.model.eval()
        self.tokenizer = llm.tokenizer
        self.generation_config = generation_config
        self.device = device

    def equivalence_prompt(self, text1: str, text2: str, question: str) -> str:
        return (
            f"""We are evaluating answers to the question "{question}"\n"""
            + f"""Here are two possible answers:\nPossible Answer 1: {text1}\nPossible Answer 2: {text2}\n"""
            + """Does Possible Answer 1 semantically entail Possible Answer 2? Respond only with entailment, contradiction, or neutral.\nResponse:"""
        )

    def predict(self, prompt: str, temperature: float) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=self.generation_config,
                temperature=temperature,
            )

        generated_ids = outputs.sequences.cpu() if hasattr(outputs, "sequences") else outputs.cpu()
        decoded = self.tokenizer.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        return decoded[0].strip()


def context_entails_response(
    context: str, responses: Sequence[str], model: BaseEntailment
) -> float:
    """Check how well responses follow from context."""
    votes = [model.check_implication(context, response) for response in responses]
    return float(2 - np.mean(votes))


def get_semantic_ids(
    strings_list: Sequence[str],
    model: BaseEntailment,
    strict_entailment: bool = False,
    example: Optional[dict] = None,
) -> List[int]:
    """Group list of predictions into semantic clusters.

    Args:
        strings_list: List of text strings to cluster
        model: Entailment model to use
        strict_entailment: If True, require bidirectional entailment
        example: Example context to use for LLM models

    Returns:
        List of cluster IDs for each input string
    """

    def are_equivalent(text1: str, text2: str) -> bool:
        implication_1 = model.check_implication(text1, text2, example=example)
        implication_2 = model.check_implication(text2, text1, example=example)
        assert all(x in [0, 1, 2] for x in [implication_1, implication_2])

        if strict_entailment:
            return (implication_1 == 2) and (implication_2 == 2)
        else:
            implications = [implication_1, implication_2]
            return (0 not in implications) and implications != [1, 1]

    semantic_set_ids = [-1] * len(strings_list)
    next_id = 0

    for i, string1 in enumerate(strings_list):
        if semantic_set_ids[i] == -1:
            semantic_set_ids[i] = next_id
            for j in range(i + 1, len(strings_list)):
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids
    return semantic_set_ids


def logsumexp_by_id(
    semantic_ids: Sequence[int],
    log_likelihoods: Sequence[float],
    agg: str = "sum_normalized",
) -> List[float]:
    """Sum probabilities with the same semantic id.

    Args:
        semantic_ids: List of cluster IDs
        log_likelihoods: Log probabilities for each item
        agg: Aggregation method ('sum_normalized' only currently supported)

    Returns:
        List of aggregated log likelihoods per semantic cluster
    """
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))

    log_likelihood_per_semantic_id = []
    log_normalizer = np.log(np.sum(np.exp(log_likelihoods)))

    for uid in unique_ids:
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]

        if agg == "sum_normalized":
            log_lik_norm = [x - log_normalizer for x in id_log_likelihoods]
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        else:
            raise ValueError(f"Unknown aggregation method: {agg}")

        log_likelihood_per_semantic_id.append(logsumexp_value)

    return log_likelihood_per_semantic_id


def predictive_entropy(log_probs: Sequence[float]) -> float:
    """Compute MC estimate of entropy.

    Args:
        log_probs: List of log probabilities

    Returns:
        Entropy estimate: -1/N sum_i log p(x_i)
    """
    return float(-np.mean(log_probs))


def predictive_entropy_rao(log_probs: Sequence[float]) -> float:
    """Compute Rao's entropy estimate.

    Args:
        log_probs: List of log probabilities

    Returns:
        Entropy estimate: -sum_i p(x_i)log p(x_i)
    """
    probs = np.exp(log_probs)
    return -np.sum(probs * log_probs)


def cluster_assignment_entropy(semantic_ids: Sequence[int]) -> float:
    """Estimate semantic uncertainty from cluster assignments.

    Args:
        semantic_ids: List of cluster IDs

    Returns:
        Entropy of the cluster assignment distribution
    """
    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts / n_generations
    assert np.isclose(probabilities.sum(), 1)
    return -np.sum(probabilities * np.log(probabilities))
