# Goldenfox Model Benchmarking 

This repo provides a pipeline to generate and evaluate summarization and prompt alignment capabilities of language models. It supports reproducible benchmarking through configurable generation and evaluation parameters.  You can choose to run the full pipeline (generation + evaluation), or just evaluation.   


To benchmark the summarization quality and prompt-following ability of models on a golden dataset of web pages with synthetic GPT-4 reference summaries.


# Key File Structure
- `benchmark.yaml` 
- `benchmark_model_pipeline.py` 
- `requirements.txt` 


# Installation Requirements 
- Create & active a fresh virtual environment 

- Install llama-cpp-python (more help at https://github.com/abetlen/llama-cpp-python)

    Linux and Mac

        CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \
        pip install llama-cpp-python


- Install remaining requirements.txt:
 `pip install -r requirements.txt` 



# How to Run
- Update `benchmark.yaml` with your desired run parameters

- Configure run parameters (To run evaluation only, set generation args to `False`)

    benchmark:
        generation:
            summarization: True
            prompt_alignment: True
        evaluation:
            summarization: True
            prompt_alignment: True


- Model Parameters:
    - `Repo`: (from huggingface)
    - `Model_name`: (the specific file)
    - `Quantization`: - the yaml file drives labeling and metadata, the model is chosen based on the model_name


- Generation is currently only supported for `gguf` format models available via HuggingFace.  

- Save changes to `benchmark.yaml`

- Run via cli command:
`python benchmark_model_pipeline.py --config benchmark.yaml`

 - Outputs will be saved under the inference/ directory for further analysis.




# Benchmarked Capabilities

- Summarization: Evaluate model’s ability to produce concise and accurate summaries.

- Prompt Alignment: Test how well the model adheres to task instructions and structured prompts.  


# Prompt Alignment 
You can add additional tasks to the prompt_alignment tests via the `golden_dataset/test/browser_tasks.yaml`. 


🧪 Evaluation Setup

    Input Data: golden_datasets/goldenfox_summarization_with_gpt_4_synthetic_reference.csv

    Run Output Directory: inference/


## System & User Prompts:





## Notes

    This benchmark limits evaluation to 50 rows for quicker iteration.

    GPT-4o summaries serve as synthetic ground truth for comparison.

    Prompt alignment is evaluated using both predefined contexts and structured browser tasks.