import json
from pathlib import Path
import time
import os
import pandas as pd
import numpy as np
import logging
import yaml
from deepeval.metrics import PromptAlignmentMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate import evaluate
from deepeval.metrics import GEval
from deepeval.metrics import SummarizationMetric
from evaluator.eval_metrics import EvaluationMetrics
from evaluator.utils.schemas import EvalJobMetrics
import argparse
from src.utils import write_to_json, read_file, setup_write_paths, load_config


def extract_evaluation_data(data, row_limit, task):
    # if row limit is 0, it means we don't need to limit at all
    row_limit = row_limit if row_limit > 0 else None

    examples = [row["example"] for row in data][0:row_limit]
    predictions = [row["prediction"] for row in data][0:row_limit]
    ground_truth = [row.get("expected_output") for row in data][0:row_limit]


    if task == 'prompt_alignment':
        # we need the prompt instructions for prompt alignment
        instructions = [row["task_metadata"].get("instructions", "none") for row in data][0:row_limit]
    else:
        instructions = ["" for row in data][0:row_limit]


    return examples, predictions, ground_truth, instructions


def evaluation_pipeline(path_to_generations, evaluation_metrics, task, row_limit):

    generations = read_file(path_to_generations)

    # for writing eval files
    output_file_path = setup_write_paths(path_to_generations)

    # initiate class
    em = EvaluationMetrics(evaluation_metrics)

    # prepare data
    examples, predictions, ground_truth, instructions = extract_evaluation_data(
        generations,
        row_limit=row_limit,
        task=task
    )

    # run eval
    evaluation_results = em.run_all(
        examples,
        predictions,
        ground_truth,
        instructions
    )

    # write to json
    write_to_json(
        evaluation_results,
        output_file_path
    )




def run_eval_pipeline(task, path_to_generations: str, row_limit: int):

    # run evaluation pipeline for summarization
    if task == "summarization":
        evaluation_metrics = ["g_eval_summarization", "rouge", "bertscore"]
        evaluation_pipeline(
            path_to_generations,
            evaluation_metrics,
            task,
            row_limit=row_limit
        )

    # run evaluation pipeline for prompt alignment
    elif task == "prompt_alignment":
        evaluation_pipeline(
            path_to_generations,
            evaluation_metrics=[task],
            task=task,
            row_limit=row_limit
        )





def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Goldenfox evaluation benchmark.")
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_run_config.yaml",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Tasks to evaluate. Overrides config if provided.",
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        default="",
        help="model name for directory",
    )

    parser.add_argument(
        "--row_limit",
        type=str,
        default=0,
        help="whether to apply a row_limit",
    )



    args = parser.parse_args(argv)
    config = load_config(args.config)

    # Fallback to config if tasks not passed on CLI
    if args.tasks:
        tasks = args.tasks
    else:
        tasks = [
            task for task, enabled in config.get("benchmark", {}).get("evaluation", {}).items()
            if enabled
        ]

    # if model_folder is not passed as an arg, use value from config
    if not args.model_folder:
        args.model_folder = config["run"]["model_name"].replace(".gguf", "")

    if not args.row_limit:
        args.row_limit = config["input_data"]["row_limit"]

    model_path = os.path.join(config["evaluation"]["run_directory"], args.model_folder)

    for task in tasks:
        path_to_generations = os.path.join(
            model_path, f"generated_{task}_{args.model_folder}.json"
        )
        run_eval_pipeline(task, path_to_generations, row_limit=args.row_limit)

if __name__ == "__main__":
    main()
