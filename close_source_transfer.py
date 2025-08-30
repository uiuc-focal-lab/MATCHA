from utils import *
from pathlib import Path
import time
import argparse
from data_utils import *

API_KEY = YOUR_API_KEY
model_libs = {
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "mistral": "mistralai/Mistral-7B-v0.1", #"/share/models/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/250544c9a802b0396550d0fd24bc80ff98bb1f5f/",
    "zephyr": "HuggingFaceH4/zephyr-7b-beta", #"/share/models/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/snapshots/b70e0c9a2d9e14bd1e812d3c398e5f313e93b473/",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama3-8B": "meta-llama/Meta-Llama-3-8B-Instruct", #"/share/models/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa/",
    "JudgeLM": "BAAI/JudgeLM-7B-v1.0",
 }

def auto_rater(args, x_0, x_p):
    prompt_rater = './autorater/criteria_prompt_close.txt'

    if x_0 == '' or x_p == '':
        return 0
    # Open the file in read mode
    with open(prompt_rater, 'r') as file:
        # Read the entire content of the file
        prompt = file.read().strip()

    # Print the content

    prompt += '\n' + 'Response 0:' + '\n' + x_0 + '\n' + 'Response 1:' + '\n' + x_p + "\n\n" + "Rating: "

    # print(prompt)
    message_list = [{"role": "user", "content": prompt}]

    # responses = chatgpt_request(model=args.model, message_list=message_list, max_tokens=args.max_length_cot, temperature=args.temperature, sleep=args.api_time_interval)
    # response = responses['choices'][0]['message']['content']

    while True:
        responses = chatgpt_request(model='gpt-3.5-turbo', message_list=message_list, max_tokens=args.max_length_cot, temperature=args.temperature, sleep=args.api_time_interval)
        response = responses['choices'][0]['message']['content']
        print(response)
        rating = response[0]

        if rating == '0' or rating == '1':
            print(f'Rating: {rating}')
            return int(rating)

def main():
    # load arguments from terminal
    args = arg_parser()
    print('*****************************')
    print(args)
    print('*****************************')

    print(f"API_KEY: {API_KEY}")

    set_random_seed(args.random_seed)

    # load dataset
    dataloader = create_dataloader_success(args)

    if args.method == "few_shot":
        input_prompt = create_input_prompt(args, cot_flag=False)
    elif args.method == "few_shot_cot" or args.method == "auto_cot" or args.method == "active_cot":
        input_prompt = create_input_prompt(args, cot_flag=True)
    else:
        raise NotImplementedError

    start = time.time()
    print("Inference Start")
    if args.multipath != 1:
        print("Self-consistency Enabled, output each inference result is not available")
    # no limit on how many batches to inference, assume inference all batches
    if args.qes_limit == 0:
        args.qes_limit = len(dataloader)

    correct, attackable, wrong_after_attack, wrong_list, QA_record = inference_cot(args, dataloader, args.qes_limit, input_prompt)
    print(f"correct: {correct}")
    print(f"total: {args.qes_limit}")
    print(f"Accuracy: {correct / (args.qes_limit)}")
    print(f"Unattackable Rate: {correct - attackable - wrong_after_attack} / {correct} = {(correct - attackable - wrong_after_attack) / correct}")
    print(f"Wrong after Attack: {wrong_after_attack} / {correct} = {wrong_after_attack / correct}")
    print(f"Attack Success Rate: {attackable} / {correct} = {(attackable) / correct}")
    end = time.time()
    print(f"Execution time: {end - start} seconds")

    print(f"wrong questions: {wrong_list}")
    # save the wrong predictions
    if args.output_dir is not None:
        path = f"{args.output_dir}/wrong_{args.dataset}.txt"
        orginal_stdout = sys.stdout
        with open(path, 'w') as f:
            sys.stdout = f
            for i in wrong_list:
                print(str(i))
        sys.stdout = orginal_stdout
        
        path = f"{args.output_dir}/QA_record_{args.dataset}.txt"
        with open(path, 'w') as f:
            f.write(json.dumps(QA_record, indent=4))


def inference_cot(args, question_pool, qes_limit, given_prompt):
    correct = 0
    qes_count = 0
    attackable = 0
    wrong_after_attack = 0
    wrong_list = []
    QA_record = []

    # judge_model_path = model_libs["JudgeLM"]
    # tokenizer_judge, model_judge = create_model(judge_model_path, None, 'cuda')

    for qes_num, qes in enumerate(question_pool):
        if qes_limit is not None and qes_count == qes_limit:
            break
        # create a list for each question to record all answers generated from self-consistency
        all_self_consistency_ans = []
        
        if args.dataset == "last_letters" and args.use_code_style_prompt == True:
            # code style prompt
            prompt = given_prompt + "Q: " + qes['question'] + "\nA: Let's think step by step in Python."
            prompt_origin = given_prompt + "Q: " + qes['origin_question'] + "\nA: Let's think step by step in Python."
        elif args.basic_cot is True:
            prompt = given_prompt + "Q: " + qes['question'] + "\nA:"
            prompt_origin = given_prompt + "Q: " + qes['origin_question'] + "\nA:"
        else:
            prompt = given_prompt + "Q: " + qes['question'] + "\nA: Let's think step by step."
            prompt_origin = given_prompt + "Q: " + qes['origin_question'] + "\nA:"
        
        # if args.model == 'gpt-3.5-turbo':
        message_list = [{"role": "user", "content": prompt}]
        message_list_origin = [{"role": "user", "content": prompt_origin}]
        # else:
            # prompt_list = [prompt]

        # enable self-consistency if multipath > 1
        for path in range(0, args.multipath):
            responses = chatgpt_request(model=args.model, message_list=message_list_origin, max_tokens=args.max_length_cot, temperature=args.temperature, sleep=args.api_time_interval)

            QA = {}
            QA['qes_idx'] = qes['question_idx']
            QA['Q'] = qes['question']
            QA['A'] = responses['choices'][0]['message']['content']
            QA_record.append(QA)

            pred_ans = answer_extraction_close(args, responses)


            content_origin = QA['A']


            # output current inference result (only works when self-consistency is not enable)
            if args.multipath == 1:
                print('-' * 20)
                print(f"Question number: {qes_num}")
                print(f"Dataset index: {qes['question_idx']}")
                print(f"Q: " + qes['question'])
                if args.dataset == "last_letters" and args.use_code_style_prompt is True:
                    print(f"A: Let's think step by step in Python." + QA['A'])
                elif args.basic_cot is True:
                    print(f"A: {QA['A']}")
                else:
                    print(f"A: Let's think step by step." + QA['A'])
                print(f"pred_ans: {pred_ans}")
                print(f"GT: {qes['answer']}")

            # record all answers into the self-consistency list to find the most frequent one
            all_self_consistency_ans.append(pred_ans)

        final_consistent_ans = find_most_frequent(all_self_consistency_ans, args.multipath)[-1]

        if final_consistent_ans == qes['answer']:
            correct += 1

            # input original question 
            responses = chatgpt_request(model=args.model, message_list=message_list, max_tokens=args.max_length_cot, temperature=args.temperature, sleep=args.api_time_interval)
            content = responses['choices'][0]['message']['content']

            pred_ans_attack = answer_extraction_close(args, responses)
            if pred_ans_attack != qes['answer']:
                wrong_after_attack += 1
                continue

            a, _ = get_answer(content_origin)
            # print(a)
            x_0_text = return_cot_close(content_origin, a)
            # print(x_0_text)

            a, _ = get_answer(content)
            # print(a)
            x_p_text = return_cot_close(content, a)
            # print(x_p_text)
            
            # use judge to detect whether the wrong reasoning
            if auto_rater(args, x_0_text, x_p_text) == 0 or check_all_equations_correct(x_p_text) == 0:
                attackable += 1
        else:
            wrong_list.append({'idx':qes['question_idx'], 'pred_ans':final_consistent_ans, 'GT':qes['answer']})

        qes_count += 1

    return correct, attackable, wrong_after_attack, wrong_list, QA_record


def arg_parser():
    parser = argparse.ArgumentParser(description="CoT")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["gsm8k","svamp", "aqua", "csqa", "asdiv", "last_letters", "addsub", "singleeq", "strategyqa", "multiarith", "time_zone", "success_list"], help="dataset to inference"
    )
    parser.add_argument(
        "--prompt_path", type=str, default="./inference_prompts/gsm8k_k=10", help="prompts to use"
    )
    # parser.add_argument(
    #     "--dataset_path", type=str, default="./results/token_attack/success_list_deepseek.json", help="dataset path to use"
    # )
    parser.add_argument(
        "--model", type=str, default="text-davinci-003", choices=["text-davinci-002", "code-davinci-002", "text-davinci-003", "gpt-3.5-turbo", "gpt-4o"], help="model used for decoding."
    )
    parser.add_argument(
        "--model2", type=str, default="llama3-8B", help="model used for decoding."
    )
    parser.add_argument(
        "--method", type=str, default="active_cot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "auto_cot", "active_cot"], help="method"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results/", help="output directory"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=256, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--qes_limit", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help="how many seconds to sleep between each request"
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help=""
    )
    parser.add_argument(
        "--multipath", type=int, default=1, help="self-consistency path num"
    )
    parser.add_argument(
        "--concat_length", type=int, default=4, help='Used for task last_letters, indicates length of last letter to concat, i.e. Elon Musk -> nk, use concat length of 2'
    )
    parser.add_argument(
        "--use_code_style_prompt", type=bool, default=False, help='Use code-style prompt as mentioned in paper for last_letters dataset'
    )
    parser.add_argument(
        "--basic_cot", type=bool, default=False, help='use basic google cot prompt of not'
    )
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    
    args = parser.parse_args()

    if args.multipath > 1:
        args.temperature = 0.7
    else:
        args.temperature = 0
    print(f"Temperature: {args.temperature}")
    
    if args.dataset == "gsm8k":
        args.dataset_path = f"./results/token_attack/success_list_gsm8k_{args.model2}_dual.txt"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "asdiv":
        args.dataset_path = "./dataset/ASDiv/ASDiv.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "The answer is"
    elif args.dataset == "csqa":
        args.dataset_path = "./dataset/CSQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "So the answer is"
    elif args.dataset == "strategyqa":
        args.dataset_path = f"./results/token_attack/success_list_strategyqa_{args.model2}_dual.txt"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters_test.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/MAWPS/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = f"./results/token_attack/success_list_singleeq_{args.model2}_dual.txt"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MAWPS/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "time_zone":
        args.dataset_path = "./dataset/timezone_convert/timezone_convertion_test.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")
        
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."
    
    return args


if __name__ == "__main__":
    main()