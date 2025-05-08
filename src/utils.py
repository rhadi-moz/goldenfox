# utils
import json
import logging
import yaml
from deepeval.test_case import LLMTestCase
import os
import numpy as np


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_to_json(text_to_write, output_path: str = None):
    def to_dict(obj):
        return getattr(obj, "dict", lambda: obj)()

    with open(output_path, "w", encoding="utf-8") as f:
        try:
            if isinstance(text_to_write, list):
                for item in text_to_write:
                    f.write(json.dumps(to_dict(item)) + "\n")
                print(f"Wrote to {output_path}")
            else:
                f.write(json.dumps(to_dict(text_to_write)) + "\n")
                print(f"Wrote to {output_path}")
            f.flush()
        except OSError as e:
            logging.info("failed to write to %s, %s", text_to_write, e)


def read_file(file_name):
    # Read from a .jsonl file
    data = []
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def setup_write_paths(input_file_path):
    return input_file_path.replace("generated_", "evaluated_")


def build_test(example, ground_truth, context_files, context_dir, task):

    llm_test_cases = []
    if len(context_files) > 0:

        for context_file in context_files:

            try:
                context_path = os.path.join(context_dir, context_file)
                with open(context_path, "r", encoding="utf-8") as f:
                    context_text = f.read()
                    context_text = context_text.strip()
            except OSError as e:
                logging.error("Failed to read context file %s: %s", context_file, e)

            # Concatenate into input prompt
            full_input = (
                f"{example.strip()} # Context {context_text}"
            )

            llm_test_case = LLMTestCase(
                input=full_input,
                expected_output=ground_truth,
                actual_output=np.nan,
                additional_metadata={
                    "category": task.get("category"),
                    "sub_category": task.get("sub_category"),
                    "context_file": context_file,
                    "instructions": example,
                },
            )

            llm_test_cases.append(llm_test_case)

    else:
            # Concatenate into input prompt
        llm_test_case = LLMTestCase(
            input=example.strip(),
            expected_output=ground_truth,
            actual_output=np.nan,
            additional_metadata={
                "category": task.get("category"),
                "sub_category": task.get("sub_category"),
                "context_file": np.nan,
                "instructions": example,
            },
        )
        llm_test_cases.append(llm_test_case)

    return llm_test_cases
