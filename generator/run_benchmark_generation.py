import os
import json
import yaml
import logging
import tqdm
import datetime
import time
import pandas as pd
from llama_cpp import Llama
import argparse
from pathlib import Path
import csv
import itertools
import gc
from src.utils import write_to_json, load_config, build_test

# --- Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class GenerateGoldenfoxBenchmark:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.context_dir = self.config['prompt_alignment']['context_file_directory']
        self.raw_browser_tasks = load_config(self.config['prompt_alignment']['task_definitions'])
        self.repo = self.config["run"]["repo"]
        self.model_name = self.config["run"]["model_name"]
        self.model_name_file = self.config["run"]["model_name"].replace('.gguf','')
        self.quantization = self.config["run"]["quantization"]
        self.temperature = self.config["generation"]["temperature"]
        self.top_k = self.config["generation"]["top_k"]
        self.top_p = self.config["generation"]["top_p"]
        self.summarization_max_tokens = self.config["summarization"]["max_tokens"]
        self.prompt_alignment_max_tokens = self.config["prompt_alignment"]["max_tokens"]
        self.min_p = self.config["generation"]["min_p"]
        self.n_ctx = self.config["generation"].get("n_ctx", 0)
        self.row_limit = self.config["input_data"]["row_limit"] if self.config["input_data"]["row_limit"] > 0 else None
        self.system_prompt = self.config["summarization"]["system"]
        self.user_prompt = self.config["summarization"]["user"]
        self.input_path = self.config["input_data"]["golden_path"]
        self.record_id = self.config["input_data"]["id_column"]
        self.n_threads = self.config["system"]["n_threads"]
        # Setup output paths
       # self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.timestamp = ''
        self.run_dir = f"inference/{self.model_name_file}{self.timestamp}"
        os.makedirs(self.run_dir, exist_ok=True)

        self.output_path = os.path.join(self.run_dir, f"{self.model_name_file}_output.json")

        # Load model
        self.llm = Llama.from_pretrained(
            repo_id=self.repo,
            filename=self.model_name,
            n_ctx=self.n_ctx,
            verbose=False,
            no_perf=False,
        )
        print("loaded model")

        self.metadata = self.llm.metadata
        print("loaded metadata")

        self.system_tokens_length = len(
            self.llm.tokenize(self.system_prompt.encode("utf-8"))
        )
        self.user_tokens_length = len(
            self.llm.tokenize(self.user_prompt.encode("utf-8"))
        )
        self.available = self.n_ctx - (
            self.system_tokens_length + self.user_tokens_length + self.summarization_max_tokens + 50
        )


    def truncate_context(self, context: str) -> str:
        return  self.llm.detokenize(
            self.llm.tokenize(context.encode("utf-8"))[0:self.available]
        ).decode("utf-8")

    def get_golden_examples(self):
        print("getting golden data")
        # load csv
        golden_set = pd.read_csv(self.input_path)

        # create lists for benchmark inputs
        ids = list(golden_set['url'])[0 : self.row_limit]
        examples = list(golden_set['text'])[0 : self.row_limit]
        ground_truths = list(golden_set['output.output'])[0 : self.row_limit]
        categories = list(golden_set['category_major'])[0 : self.row_limit]

        return ids, examples, ground_truths, categories

    def browser_tasks(self):
        browser_tasks = []

        for task in self.raw_browser_tasks:

            browser_task = build_test(
                example=task['user_prompt'],
                ground_truth=task.get("expected_output") or '',
                context_files = task.get('context_files',[]),
                context_dir=self.context_dir,
                task=task
            )
            browser_tasks.extend(browser_task)


        return browser_tasks[0:self.row_limit]

    def build_messages(
        self,
        system_prompt: str = None,
        user_prompt: str = None,
        context: str = ""
    ) -> list:

        # Fallback to instance defaults if parameters not provided
        system_prompt = system_prompt or self.system_prompt
        user_prompt = user_prompt or self.user_prompt
        context = context or ""

        # truncate to available tokens (if less than available tokens, will include full context)
        context = self.truncate_context(context)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_prompt}{context}"},
        ]

    def predict(
        self,
        input_id: str = None,
        system_prompt: str = None,
        user_prompt: str = None,
        context: str = None,
        max_tokens: int = None,
    ) -> dict:

        # reset llm to get rid of cache from any prior generations
        self.llm.reset()

        # generation parameters
        max_tokens = max_tokens or self.summarization_max_tokens
        system_prompt = system_prompt or self.system_prompt
        user_prompt = user_prompt or self.user_prompt

        print("calling model")

        # build messages (applies truncation if needed)
        messages = self.build_messages(
            system_prompt,
            user_prompt,
            context
        )

        # start timer
        start = datetime.datetime.now()

        # call llm
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
            top_k=self.top_k,
            top_p=self.top_p,
            min_p=self.min_p,
        )

        # end timer
        end = datetime.datetime.now()

        # return result with metadata
        return {
            "input_id": input_id,
            "example": user_prompt,
            "context": context,
            "prediction": response["choices"][0]["message"]["content"],
            "usage": response["usage"],
            "latency_sec": (end - start).total_seconds(),
            "tokens_per_second": (
                response["usage"].get("completion_tokens")
                / (end - start).total_seconds()
                if (end - start).total_seconds() > 0
                else 0
            ),
        }

    def run_summarization_benchmark(self, task):
        output_path = Path(f"generated_{task}_{self.model_name_file}{self.timestamp}.json")
        ids, examples, ground_truths, categories = self.get_golden_examples()

        results = []

        for id_, example, ground_truth, category in zip(ids, examples, ground_truths, categories):
            print(id_)
            output = self.predict(
                id_,
                user_prompt=self.user_prompt,
                context=example
            )
            output['expected_output'] = ground_truth
            output['category'] = category
            results.append(output)

        write_to_json(results, os.path.join(self.run_dir, output_path))


    def run_browser_task(self, browser_task):
        result = self.predict(
            user_prompt=browser_task.input,
            max_tokens=self.prompt_alignment_max_tokens
        )
        result["task_metadata"] = browser_task.additional_metadata
        result["expected_output"] = browser_task.expected_output
        return result


    def run_browser_task_benchmark(self, task):
        browser_test_cases = self.browser_tasks()

        output_path = Path(f"generated_{task}_{self.model_name_file}{self.timestamp}.json")

        task_results = []
        for browser_task in browser_test_cases:
            task_result = self.run_browser_task(browser_task)
            task_results.append(task_result)

        write_to_json(task_results, os.path.join(self.run_dir, output_path))


    def save_run_metadata(self):
        os.makedirs(self.run_dir, exist_ok=True)

        llama_metadata = self.llm.metadata or {}

        metadata = {
            "timestamp": self.timestamp,
            "model_metadata": self.llm.metadata,
            "model_name": self.config["run"].get("model_name").replace(".gguf",""),
            "model_repo": self.config["run"].get("repo"),
            "quantization": self.config["run"].get("quantization"),
            "temperature": self.config["generation"].get("temperature"),
            "top_k": self.config["generation"].get("top_k"),
            "top_p": self.config["generation"].get("top_p"),
            "min_p": self.config["generation"].get("min_p"),
            "n_ctx": self.config["generation"].get("n_ctx"),
            "n_threads": self.config["system"].get("n_threads"),
            "summarization_max_tokens": self.config["summarization"].get("max_tokens"),
            "prompt_alignment_max_tokens": self.config["prompt_alignment"].get("max_tokens"),
            "model_size": llama_metadata.get("general.size_label"),

        }

        # Save to JSON
        json_path = os.path.join(self.run_dir, f"{metadata['model_name']}_metadata.json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {json_path}")




def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Goldenfox model benchmark.")
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_run_config.yaml",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Tasks to evaluate (e.g. summarization prompt_alignment). If not provided, uses config file.",
    )

    args = parser.parse_args(argv)
    config = load_config(args.config)

    # Determine tasks to run
    if args.tasks:
        tasks_to_run = set(args.tasks)
    else:
        # Fallback to config structure: benchmark: { task_name: bool }
        benchmark_flags = config['benchmark']['generation']
        tasks_to_run = [task for task, enabled in benchmark_flags.items() if enabled]

    # Initialize benchmark runner
    benchmark = GenerateGoldenfoxBenchmark(config_path=args.config)

    # Run selected tasks
    for task in tasks_to_run:

        if task == 'summarization':
            print(f"Running: {task}")
            benchmark.run_summarization_benchmark(task=task)

        elif task == 'prompt_alignment':
            print(f"Running: {task}")
            benchmark.run_browser_task_benchmark(task=task)

    # save run metadata for reporting
    benchmark.save_run_metadata()



if __name__ == "__main__":
    main()
