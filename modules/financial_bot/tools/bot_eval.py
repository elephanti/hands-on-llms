import logging
import json

import fire

from ragas.metrics import (
    answer_correctness,
    answer_similarity,
    #context_entity_recall,
    context_recall,
    #context_relevancy,
    #context_utilization,
    faithfulness
)
from ragas.metrics.context_precision import context_relevancy
from ragas import evaluate
from datasets import Dataset

from tools.bot import load_bot

logger = logging.getLogger(__name__)

METRICS = [
    #context_utilization,
    context_relevancy,
    context_recall,
    answer_similarity,
    #context_entity_recall,
    #answer_correctness,
    faithfulness
]

def evaluate_w_ragas(query: str, context: list[str], output: str, ground_truth: str) -> dict:
    """
    Evaluate the RAG (query,context,response) using RAGAS
    """
    data_sample = {
        "question": [query],  # Question as Sequence(str)
        "answer": [output],  # Answer as Sequence(str)
        "contexts": [context],  # Context as Sequence(str)
        "ground_truths": [[ground_truth]],  # Ground Truth as Sequence(str)
    }

    dataset = Dataset.from_dict(data_sample)
    score = evaluate(
        dataset=dataset,
        metrics=METRICS,
    )

    return score

def run_local(
    testset_path: str,
):
    """
    Run the bot locally in production or dev mode.

    Args:
        testset_path (str): A string containing path to the testset.

    Returns:
        str: A string containing the bot's response to the user's question.
    """

    bot = load_bot(model_cache_dir=None)

    from financial_bot import utils

    logger.info("#" * 100)
    utils.log_available_gpu_memory()
    utils.log_available_ram()
    logger.info("#" * 100)

    with open(testset_path, "r") as f:
        data = json.load(f)
        for elem in data:
            input_payload = {
                "about_me": elem["about_me"],
                "question": elem["question"],
                "to_load_history": [],
            }
            output_context = bot.finbot_chain.chains[0].run(input_payload)
            response = bot.answer(**input_payload)
            logger.info("Score=%s", evaluate_w_ragas(query=elem["question"], context=output_context.split('\n'), output=response, ground_truth=elem["response"]))

    return response


if __name__ == "__main__":
    fire.Fire(run_local)
