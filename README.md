# MATCHA
Misaligning Reasoning with Answers - A Framework for Assessing LLM CoT Robustness[[Arxiv]](https://arxiv.org/abs/2505.17406)\
*Enyi Jiang, Changming Xu, Nischay Singh, Gagandeep Singh*


## Code

### Installation
We recommend first creating a conda environment using the provided [requirements.txt](https://github.com/uiuc-focal-lab/MATCHA/blob/main/requirements.txt):

`conda create --name MATCHA`

`pip install -r requirements.txt`

### Testing

### Run embedding-level perturbation script
```shell
python inference_emb_attack.py --dataset="gsm8k" --model="llama3-8B" --method="few_shot_cot" --qes_limit=0 --prompt_path="./basic_cot_prompts/math_word_problems" --random_seed=42 --multipath=1 --basic_cot True
```

### Run token-level perturbation script
```shell
python inference_tok_random_dual.py --dataset="gsm8k" --model="llama3-8B" --method="few_shot_cot" --qes_limit=0 --prompt_path="./basic_cot_prompts/math_word_problems" --random_seed=42 --multipath=1 --basic_cot True
```

## Important arguments
   * `--dataset`: The name of a dataset. `choices = [gsm8k, svamp, aqua, csqa, last_letters, strategyqa, asdiv, singleeq, addsub, multiarith]`.
   * `--model`: open-source model. `choices = [["llama3-8B","mistral", "zephyr", "qwen", "deepseek"]`.
   * `--method`: few-shot-cot or active_cot.
   * `--qes_limit`: number of test questions.
   * `--prompt_path`: path of prompt file.



## Credits
Parts of the code in this repo is based on
+ [https://github.com/shizhediao/active-prompt](https://github.com/shizhediao/active-prompt)

## Citation
Cite the paper/repo:
```
@article{jiang2025misaligning,
  title={Misaligning Reasoning with Answers--A Framework for Assessing LLM CoT Robustness},
  author={Jiang, Enyi and Xu, Changming and Singh, Nischay and Singh, Gagandeep},
  journal={arXiv preprint arXiv:2505.17406},
  year={2025}
}
```

