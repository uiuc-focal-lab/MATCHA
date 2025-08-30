# This file contains necessary helper functions
# e.g. GPT request, create_dataloader
import openai
import random
import sys
import numpy as np
import torch
import json
import re
from collections import Counter
import time
from sympy import sympify
import ast
import os
os.environ['HF_HOME'] = None

from typing import Tuple, Optional, Any, List
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,
                          GPTJForCausalLM, GPTNeoXForCausalLM,
                          LlamaForCausalLM, MistralForCausalLM, 
                          Qwen2ForCausalLM)

# put your API key here
API_KEY = YOUR_API_KEY
# define for no solution if GPT cannot generate a valid solution
# here define a magic number for the convenience of variance calculatio
NO_SOLUTION = '-10086' # use this when calculating numerical results

# set the random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def chatgpt_request(model:str, message_list:list, max_tokens:int, temperature=0.7, sleep=3):
    resp = None
    done = False
    count = 0
    while not done:
        if count > 10:
            return None
        try:
            openai.api_key = API_KEY
            resp = openai.ChatCompletion.create(
                model=model,
                messages=message_list,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1.0,
            )
            done = True
        except:
            errno = sys.exc_info()[:2]
            if errno[0] == openai.error.InvalidRequestError:
                # print(f"Invalid Request\nPrompt: {message_list}\n")
                print("Invalid Request")
                print(f"Reason: {errno[1]}")
                assert False
            else:
                print(f"Error: {errno[0]}\n")
                print(f"Reason: {errno[1]}\n")
            # pause between each request to avoid rate limit
            time.sleep(sleep)
            count += 1
    return resp


# pass in a list of prompts and returns a response body contains a list of responses
def GPT3_request(model:str, input_prompt:list, max_tokens:int, time_interval, temperature=0.7, stop=None):
    resp = None
    done = False
    while not done:
        try:
            openai.api_key = API_KEY
            resp = openai.Completion.create(
                model=model,
                prompt=input_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop = stop
            )
            done = True
        except:
            errno = sys.exc_info()[:2]
            if errno[0] == openai.error.InvalidRequestError:
                print(f"Invalid Request\nPrompt: {input_prompt}\n")
                print(f"Reason: {errno[1]}")
                assert False
            else:
                print(f"Error: {errno[0]}\n")
                print(f"Reason: {errno[1]}\n")
            # pause between each request to avoid rate limit
            time.sleep(time_interval)
    return resp


def load_data(args):
    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset == "gsm8k":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(json_res["answer"].split("#### ")[-1].replace(",", ""))
    elif args.dataset == "aqua":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                qes = json_res["question"].strip() + " Answer Choices:"

                for opt in json_res["options"]:
                    opt = opt.replace(')', ') ')
                    qes += f" ({opt}"

                questions.append(qes)
                answers.append(json_res["correct"])
    elif args.dataset == "svamp":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)
    elif args.dataset == "asdiv":
        with open(args.dataset_path) as f:
            json_data = json.load(f)["Instances"]
            for line in json_data:
                q = line['input'].strip()
                a = line['output'][0]
                questions.append(q)
                answers.append(a)
    elif args.dataset in ("addsub", "singleeq", "multiarith"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["sQuestion"].strip()
                a = str(line["lSolutions"][0])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)
    elif args.dataset == "csqa":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "Answer Choices:"
                for c in json_res["question"]["choices"]:
                    choice += " ("
                    choice += c["label"]
                    choice += ") "
                    choice += c["text"]
                questions.append(json_res["question"]["stem"].strip() + " " + choice)
                answers.append(json_res["answerKey"])
    elif args.dataset == "strategyqa":
        if 'task' in args.dataset_path:
            with open(args.dataset_path) as f:
                json_data = json.load(f)["examples"]
                for line in json_data:
                    q = line["input"].strip()
                    a = int(line["target_scores"]["Yes"])
                    if a == 1:
                        a = "yes"
                    else:
                        a = "no"
                    questions.append(q)
                    answers.append(a)
        else:
            with open(args.dataset_path, encoding='utf-8') as f:
                json_data = json.load(f)
                for line in json_data:
                    q = line["question"].strip() 
                    if line['answer']:
                        a = 'yes'
                    else:
                        a = 'no'
                    questions.append(q)
                    answers.append(a)
    elif args.dataset in ("coin_flip", "last_letters"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            json_data = json_data["examples"]
            for line in json_data:
                q = line["question"]
                a = line["answer"]
                questions.append(q)
                answers.append(a)
    elif args.dataset == 'time_zone':
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line['question'].strip()
                a = line["answer"]
                questions.append(q)
                answers.append(a)
    else:
        raise NotImplementedError

    print(f"dataset: {args.dataset}")
    print(f"dataset_size: {len(answers)}")
    args.dataset_size = len(answers)
    return questions, answers

def load_success_lst(args):
    questions = []
    origin_questions = []
    answers = []
    with open(args.dataset_path) as f:
        lines = f.readlines()
        for line in lines:
            result_dict = ast.literal_eval(line)
            q = result_dict['new_question'].strip()
            q_origin = result_dict['old_question'].strip()
            a = result_dict["answer"]
            questions.append(q)
            origin_questions.append(q_origin)
            answers.append(a)
    
    print(f"dataset: {args.dataset}")
    print(f"dataset_size: {len(answers)}")
    args.dataset_size = len(answers)

    return questions, origin_questions, answers
# return a customized dataloader of batches
# Not PyTorch dataloader, it supprts random index(slice) access
def create_dataloader(args)->list:
    set_random_seed(args.random_seed)
    questions, answers = load_data(args)
    dataset = []
    for idx in range(len(questions)):
        dataset.append({"question":questions[idx], "answer":answers[idx], "question_idx":idx})

    random.shuffle(dataset)
    print(f"dataloader size: {len(dataset)}")
    return dataset

def create_dataloader_success(args)->list:
    set_random_seed(args.random_seed)
    questions, origin_questions, answers = load_success_lst(args)
    dataset = []
    for idx in range(len(questions)):
        dataset.append({"question":questions[idx], "origin_question":origin_questions[idx], "answer":answers[idx], "question_idx":idx})

    random.shuffle(dataset)
    print(f"dataloader size: {len(dataset)}")
    return dataset


# read the generated/prepared prompt json file
# return a string of prefix prompt before each question
def create_input_prompt(args, cot_flag:bool)->str:
    x, z, y = [], [], []
    
    with open(args.prompt_path, encoding="utf-8") as f:
        json_data = json.load(f)
        json_data = json_data["prompt"]
        for line in json_data:
            x.append(line["question"])
            z.append(line["rationale"])
            y.append(line["pred_ans"])

    index_list = list(range(len(x)))
    
    prompt_text = ""
    for i in index_list:
        if cot_flag:
            if args.dataset == "strategyqa":\
                # first answer, then reasoning
                # prompt_text += x[i] + " " + z[i] + " " + \
                #             "So the answer is" + " " + y[i] + ".\n\n"
                prompt_text += x[i] + " " + args.direct_answer_trigger_for_fewshot + " " \
                    + y[i] + " " + z[i] + ".\n\n"
            else:
                # first answer, then reasoning
                # prompt_text += x[i] + " " + z[i] + " " + \
                #             args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
                prompt_text += x[i] + " " + args.direct_answer_trigger_for_fewshot + " " \
                    + y[i] + " " + z[i] + ".\n\n"
        else:
            prompt_text += x[i] + " " + args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
    return prompt_text

# return a string of prefix prompt before each question
def create_input_prompt_reason(args, cot_flag:bool)->str:
    x, z, y = [], [], []
    
    with open(args.prompt_path, encoding="utf-8") as f:
        json_data = json.load(f)
        json_data = json_data["prompt"]
        for line in json_data:
            x.append(line["question"])
            z.append(line["rationale"])
            y.append(line["pred_ans"])

    index_list = list(range(len(x)))
    
    prompt_text = ""
    for i in index_list:
        if cot_flag:
            if args.dataset == "strategyqa":\
                # first reasoning, then answer
                prompt_text += x[i] + " " + z[i] + " " + \
                            "So the answer is" + " " + y[i] + ".\n\n"
                # prompt_text += x[i] + " " + args.direct_answer_trigger_for_fewshot + " " \
                #     + y[i] + " " + z[i] + ".\n\n"
            else:
                # first ansreasoningwer, then reasoning
                prompt_text += x[i] + " " + z[i] + " " + \
                            args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
                # prompt_text += x[i] + " " + args.direct_answer_trigger_for_fewshot + " " \
                #     + y[i] + " " + z[i] + ".\n\n"
        else:
            prompt_text += x[i] + " " + args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
    return prompt_text

def answer_extraction(args, responses):
    pred_ans = ""
    temp = ""
    # if args.model == 'gpt-3.5-turbo':
    #     temp = responses['choices'][0]['message']['content']
    # else:
    #     temp = responses['choices'][0].text
    temp = responses
    if args.dataset in ("gsm8k", "svamp", "asdiv", "addsub", "singleeq", "multiarith"):
        temp = temp.replace(",", "")
        temp = [s for s in re.findall(r'-?\d+\.?\d*', temp)]
    elif args.dataset in ("aqua", "csqa"):
        temp = re.findall(r'A|B|C|D|E', temp)
    elif args.dataset in ("strategyqa", "coin_flip"):
        temp = temp.lower()
        temp = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", temp)
        temp = temp.split(" ")
        temp = [i for i in temp if i in ("yes", "no")]
    elif args.dataset in ("last_letters"):
        temp = re.sub("\"|\'|\n|\.|\s","", temp)
        temp = [temp]
    elif args.dataset in ('time_zone'):
        temp = temp.split('The answer is ')[-1].replace('.', '')
        temp = [temp]

    if len(temp) != 0:
        # answer = temp[-1]
        answer = temp[0]
        # if there is . at the end of answer, remove it
        # e.g. answer = 64.
        if answer != "":
            if answer[-1] == ".":
                answer = answer[:-1]

        # round the answer to nearest integer
        if args.dataset in ("gsm8k", "svamp"):
            try:
                answer = str(round(float(answer)))
            except:
                answer = "" # no sol or sol doesn't have valid format
        elif args.dataset in ("last_letters"):
            try:
                answer = answer[-args.concat_length:]
            except:
                answer = ""
        pred_ans = answer
    else:
        pred_ans = ""
    
    if args.dataset == 'math':
        _, pred_ans = get_answer_math(responses)
    return pred_ans

def answer_extraction_close(args, responses):
    pred_ans = ""
    temp = ""
    # if args.model == 'gpt-3.5-turbo':
    temp = responses['choices'][0]['message']['content']
    # else:
        # temp = responses['choices'][0].text
    if args.dataset in ("gsm8k", "svamp", "asdiv", "addsub", "singleeq", "multiarith"):
        temp = temp.replace(",", "")
        temp = [s for s in re.findall(r'-?\d+\.?\d*', temp)]
    elif args.dataset in ("aqua", "csqa"):
        temp = re.findall(r'A|B|C|D|E', temp)
    elif args.dataset in ("strategyqa", "coin_flip"):
        temp = temp.lower()
        temp = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", temp)
        temp = temp.split(" ")
        temp = [i for i in temp if i in ("yes", "no")]
    elif args.dataset in ("last_letters"):
        temp = re.sub("\"|\'|\n|\.|\s","", temp)
        temp = [temp]
    elif args.dataset in ('time_zone'):
        temp = temp.split('The answer is ')[-1].replace('.', '')
        temp = [temp]

    if len(temp) != 0:
        answer = temp[-1]
        # if there is . at the end of answer, remove it
        # e.g. answer = 64.
        if answer != "":
            if answer[-1] == ".":
                answer = answer[:-1]

        # round the answer to nearest integer
        if args.dataset in ("gsm8k", "svamp"):
            try:
                answer = str(round(float(answer)))
            except:
                answer = "" # no sol or sol doesn't have valid format
        elif args.dataset in ("last_letters"):
            try:
                answer = answer[-args.concat_length:]
            except:
                answer = ""
        pred_ans = answer
    else:
        pred_ans = ""
    return pred_ans

def answer_extraction_reason(args, responses):
    pred_ans = ""
    temp = ""
    # if args.model == 'gpt-3.5-turbo':
    #     temp = responses['choices'][0]['message']['content']
    # else:
    #     temp = responses['choices'][0].text
    temp = responses
    if('answer is ' in temp):
        temp = temp.split('answer is ')[1]
    if args.dataset in ("gsm8k", "svamp", "asdiv", "addsub", "singleeq", "multiarith"):
        temp = temp.replace(",", "")
        temp = [s for s in re.findall(r'-?\d+\.?\d*', temp)]
    elif args.dataset in ("aqua", "csqa"):
        temp = re.findall(r'A|B|C|D|E', temp)
    elif args.dataset in ("strategyqa", "coin_flip"):
        temp = temp.lower()
        temp = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", temp)
        temp = temp.split(" ")
        temp = [i for i in temp if i in ("yes", "no")]
    elif args.dataset in ("last_letters"):
        temp = re.sub("\"|\'|\n|\.|\s","", temp)
        temp = [temp]
    elif args.dataset in ('time_zone'):
        temp = temp.split('The answer is ')[-1].replace('.', '')
        temp = [temp]

    if len(temp) != 0:
        answer = temp[0]
        # if there is . at the end of answer, remove it
        # e.g. answer = 64.
        if answer != "":
            if answer[-1] == ".":
                answer = answer[:-1]

        # round the answer to nearest integer
        if args.dataset in ("gsm8k", "svamp"):
            try:
                answer = str(round(float(answer)))
            except:
                answer = "" # no sol or sol doesn't have valid format
        elif args.dataset in ("last_letters"):
            try:
                answer = answer[-args.concat_length:]
            except:
                answer = ""
        pred_ans = answer
    else:
        pred_ans = ""
    return pred_ans


def find_most_frequent(arr, n):
    # method 1: return max(arr[:n], key=arr.count)
    # method 2:
    arr_acounts = Counter(arr[:n])
    most_frequent_item, frequency = arr_acounts.most_common(1)[0]
    return frequency, most_frequent_item


def create_model(model_name: str, peft_model_name: Optional[str], device: str,
                  do_compile: bool = True, dtype: torch.dtype = torch.bfloat16) -> Tuple[Any, Any]:

    model_kwargs = {}
    peft_kwargs = {}

    if device == "cuda":
        model_kwargs['torch_dtype'] = peft_kwargs['torch_dtype'] = dtype
        model_kwargs['device_map'] = peft_kwargs['device_map'] =  'auto'  # 'auto'
    else:
        model_kwargs['low_cpu_mem_usage'] = True

    tokenizer = AutoTokenizer.from_pretrained(model_name, resume_download=True, padding_side="left")
    if model_name == 'qwen':
        tokenizer.pad_token = '<|endoftext|>'
        tokenizer.eos_token = '<|endoftext|>'
    else:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, resume_download=True, trust_remote_code=True, **model_kwargs)

    # if do_compile is True:
    #     model = torch.compile(model)

    model.eval()

    return tokenizer, model

def add_radius_noise_to_embedding(embedding, radius):
    """
    Modifies the input embedding by adding noise with a specified fixed radius
    in a random direction.
    
    Parameters:
    embedding (torch.Tensor): The input sentence embedding (1D tensor).
    radius (float): The fixed radius of the noise.
    
    Returns:
    torch.Tensor: The modified embedding.
    """
    
    # Get the dimensionality of the embedding
    dim = embedding.shape[2]
    
    # Sample a random direction uniformly from a unit sphere
    random_direction = torch.randn(dim, dtype=embedding.dtype)
    
    # Normalize the direction to get a unit vector
    random_direction = random_direction / torch.norm(random_direction, p=2)
    
    # Scale the random direction by the fixed radius
    noise = (radius * random_direction).cuda()

    # print(noise.shape)
    
    # Add the noise to the original embedding
    modified_embedding = embedding + noise
    
    return modified_embedding

def get_embeddings(model, input_ids):
    return model.model.embed_tokens(input_ids)

def get_embedding_matrix(model):
    # if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
    #     return model.transformer.wte.weight
    if isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens.weight
    elif isinstance(model, MistralForCausalLM):
        return model.model.embed_tokens.weight
    elif isinstance(model, Qwen2ForCausalLM):
        return model.model.embed_tokens.weight
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in.weight
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def get_answer(input):
    if('The answer is ' in input):
        lines = input.split('The answer is ')[1].split(' ')[0]
    else:
        lines = ""
    return 'The answer is ' + lines, lines

def get_answer_math(input):
    if('The answer is ' in input):
        lines = input.split('The answer is ')[1].split('\n')[0]
    else:
        pattern = '\d*\.?\d+'
        pred = re.findall(pattern, input)
        if(len(pred) >= 1):
            # print(pred_str)
            pred = pred[0]
        else: 
            pred = ''
        lines = pred
    return 'The answer is ' + lines, lines

def get_rating(input):
    if('Rating: ' in input):
        rating = input.split('Rating: ')[1][0]
    else:
        rating = None
    return rating

def return_cot(input, a):
    if('The answer is ' in input):
        try: 
            lines = input.split(a)[1].split('Q:')[0].strip()
        except Exception:
            lines = input.strip()
    else:
        lines = input.strip()
    return lines

def return_cot_close(input, a):
    if('The answer is ' in input):
        try: 
            lines = input.split(a)[1].strip()
        except Exception:
            lines = input.strip()
    else:
        lines = input.strip()
    return lines

def return_cot_first(input):
    if('The answer is ' in input):
        lines = input.split('The answer is ')[0].strip()
    else:
        lines = input.strip()
    return lines

def return_cot_2(input):
    lines = input.split('Q:')[0].strip()
    return lines

def find_answer(s):
    assert('boxed' in s)
    ans = s.split('boxed')[-1]
    if(ans[0] == '{'):
        stack = 1
        a = ''
        for c in ans[1:]:
            if(c == '{'): 
                stack += 1
                a += c
            elif(c == '}'): 
                stack -= 1
                if(stack == 0): break
                a += c
            else: 
                a += c
    else:
        a = ans.split('$')[0].strip()
    return a


def check_all_equations_correct(text):

    text = text.replace(',', '')
    # Regex pattern to capture equations of the form "a x b = c"
    # equation_pattern = r"((\d+\s*[\+\-\*/xX])*(\s*\d+\s*)=\s*\d+)"
    # equation_pattern = r"(\d+\s*[\+\-\*/xX]\s*\d+\s*=\s*\d+)"
    # equation_pattern = r'\d+\.?\d*(?:\s*[+*x/÷\-]\s*\d+\.?\d*)+\s*=\s*\d+\.?\d*'
    # equation_pattern = r"(\d+(?:\s*[+\-*/xX]\s*\d+)*\s*=\s*\d+)"
    equation_pattern = r'\d+(?:\.\d*)?(?:\s*[+*x/÷\-]\s*\d+(?:\.\d*)?)+\s*=\s*\d+(?:\.\d*)?(?:\s*/\s*\d+(?:\.\d*)?)?'
    
    # Find all equations in the string
    equations = re.findall(equation_pattern, text)

    print(equations)
    
    for eq in equations:
        # Normalize "x" to "*" for multiplication
        eq_normalized = eq.replace('x', '*').replace('X', '*')

        if eq_normalized.endswith('.'):
            eq_normalized = eq_normalized[:-1]
        
        # Split the equation into left-hand side and right-hand side
        lhs, rhs = eq_normalized.split('=')
        
        # Evaluate the left-hand side
        try:
            lhs_value = float(sympify(lhs.strip()))
            rhs_value = float(rhs.strip())

            # print(lhs_value, rhs_value)
            
            # Check if the equation is incorrect
            if abs(lhs_value - rhs_value) > 1e-2:
                print('wrong eq:', lhs, rhs)
                return 0  # If any equation is incorrect, return 0
        except Exception as e:
            print(e)
            # return 0  # If evaluation fails, return 0
            continue
    
    return 1  # Return 1 only if all equations are correct

def remove_repeated_phrases(text):
    # Use a lambda to replace each repeated occurrence with the first match
    cleaned_text = re.sub(r'(\b[\w\s.,\'"]+?[.!?])(?:\s*\1)+', r'\1', text)
    return cleaned_text