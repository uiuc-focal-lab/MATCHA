from utils import *
from pathlib import Path
import time
import argparse
from data_utils import *
from torch.autograd import Variable
import torch.nn as nn
import os
import gc
from copy import deepcopy
import re
import random
from sympy import sympify
import sys
# from inference_token_attack_dual_opt import *



model_libs = {
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "mistral": "mistralai/Mistral-7B-v0.1", #"/share/models/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/250544c9a802b0396550d0fd24bc80ff98bb1f5f/",
    "zephyr": "HuggingFaceH4/zephyr-7b-beta", #"/share/models/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/snapshots/b70e0c9a2d9e14bd1e812d3c398e5f313e93b473/",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama3-8B": "meta-llama/Meta-Llama-3-8B-Instruct", #"/share/models/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa/",
    "JudgeLM": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", #"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",#"BAAI/JudgeLM-7B-v1.0",
    "JedgeLM2": "BAAI/JudgeLM-7B-v1.0"
 }



def main():
    # load arguments from terminal
    args = arg_parser()

    print('*****************************')
    print(args)
    print('*****************************')

    # print(f"API_KEY: {API_KEY}")

    set_random_seed(args.random_seed)

    # load dataset
    dataloader = create_dataloader(args)

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

    correct, attackable, wrong_after_attack, avg_length, wrong_list, success_list, QA_record = inference_cot_token_attack(args, dataloader, args.qes_limit, input_prompt)
    print(f"correct: {correct}")
    print(f"total: {args.qes_limit}")
    print(f"Accuracy: {correct / (args.qes_limit)}")
    print(f"Unattackable Rate: {correct - attackable - wrong_after_attack} / {correct} = {(correct - attackable - wrong_after_attack) / correct}")
    print(f"Wrong after Attack: {wrong_after_attack} / {correct} = {wrong_after_attack / correct}")
    print(f"Attack Success Rate: {attackable} / {correct} = {(attackable) / correct}")
    print(f"Average attack steps: {avg_length}")
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

        path = f"{args.output_dir}/success_list_{args.dataset}_{args.model}_dual.txt"
        # with open(path, 'w') as f:
        #     f.write(json.dumps(success_list, indent=4))
        # orginal_stdout = sys.stdout
        # with open(path, 'w') as f:
        #     sys.stdout = f
        #     for i in success_list:
        #         print(str(i))
        # sys.stdout = orginal_stdout


def return_response(args, input_emb, attention_mask, max_len, tokenizer, model):
    with torch.inference_mode():
        generate_ids = model.generate(inputs_embeds=input_emb, 
                        attention_mask=attention_mask,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                        max_new_tokens=max_len, do_sample=False, temperature=args.temperature)
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response

def get_inserted_locations(original_ids, perturbed_ids):
    """
    Compare original and perturbed input_ids to find inserted token locations.
    Returns a list of indices where tokens were inserted.
    """
    inserted_locations = []
    orig_len = original_ids.shape[1]
    perturb_len = perturbed_ids.shape[1]
    
    # Handle case where lengths are different
    if orig_len != perturb_len:
        i, j = 0, 0
        while i < orig_len and j < perturb_len:
            if original_ids[0][i] != perturbed_ids[0][j]:
                inserted_locations.append(j)
                j += 1
            else:
                i += 1
                j += 1
        # Add remaining positions if perturbed sequence is longer
        while j < perturb_len:
            inserted_locations.append(j)
            j += 1
    
    return inserted_locations

def token_gradients_x(args, model, input_ids, target_logits, loss, length_w_answer, length_w_cot):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    target_logits: torch.Tensor
        target logit distribution

    Returns
    -------
    torch.Tensor
        The gradients of each token with respect to the loss.
    """
    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids.shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    # print(one_hot.shape, input_ids.shape, embed_weights.shape)
    one_hot.scatter_(
        1, 
        input_ids.unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    logits = model(inputs_embeds=input_embeds).logits
    
    cost = - loss(logits[:, length_w_answer:length_w_cot, :], target_logits[:, length_w_answer:length_w_cot, :])
    # + args.lbd * loss(logits[:, :length_w_answer, :], target_logits[:, :length_w_answer, :])
    
    # cost.backward(retain_graph=True)
    print(cost)
    grad = torch.autograd.grad(
            cost, one_hot, retain_graph=False, create_graph=False, allow_unused=True
        )[0]
    
    return grad.clone()

def token_gradients_answer(args, model, input_ids, target_logits, loss):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    target_logits: torch.Tensor
        target logit distribution

    Returns
    -------
    torch.Tensor
        The gradients of each token with respect to the loss.
    """
    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids.shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    # print(one_hot.shape, input_ids.shape, embed_weights.shape)
    one_hot.scatter_(
        1, 
        input_ids.unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    logits = model(inputs_embeds=input_embeds).logits
    
    cost = loss(logits, target_logits)
    
    # cost.backward(retain_graph=True)
    print(cost)
    grad = torch.autograd.grad(
            cost, one_hot, retain_graph=False, create_graph=False, allow_unused=True
        )[0]
    
    return grad.clone()

def token_gradients_both(args, model, input_ids, target_logits, loss, length_w_answer):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    target_logits: torch.Tensor
        target logit distribution

    Returns
    -------
    torch.Tensor
        The gradients of each token with respect to the loss.
    """
    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids.shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    # print(one_hot.shape, input_ids.shape, embed_weights.shape)
    one_hot.scatter_(
        1, 
        input_ids.unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    logits = model(inputs_embeds=input_embeds).logits
    
    # print(loss(logits[:, length_w_answer:length_w_cot, :], target_logits[:, length_w_answer:length_w_cot, :]))
    # print(loss(logits[:, :length_w_answer, :], target_logits[:, :length_w_answer, :]))
    #  + 
    # print(target_logits.shape[1])
    cost = - loss(logits[:, length_w_answer:, :], target_logits[:, length_w_answer:, :]) + ((target_logits.shape[1] - length_w_answer) / length_w_answer) * loss(logits[:, :length_w_answer, :], target_logits[:, :length_w_answer, :])
    
    # cost.backward(retain_graph=True)
    print(cost)
    grad = torch.autograd.grad(
            cost, one_hot, retain_graph=False, create_graph=False, allow_unused=True
        )[0]
    
    return grad.clone()


def auto_rater(x_0, x_p, max_len, tokenizer, model):
    prompt_rater = './autorater/criteria_prompt.txt'

    # Open the file in read mode
    with open(prompt_rater, 'r') as file:
        # Read the entire content of the file
        prompt = file.read().strip()

    # Print the content

    prompt += '\n' + 'Response 0:' + '\n' + x_0 + '\n' + 'Response 1:' + '\n' + x_p + "\n\n" + "Rating: "

    # print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt")
        
    for key in inputs:
        inputs[key] = inputs[key].cuda()
    
    while True:
        with torch.inference_mode():
            generate_ids = model.generate(**inputs, max_new_tokens=max_len, do_sample=False)
        
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        rating = get_rating(response)
        # print(response)

        if rating:
            print(f'Rating: {rating}')
            return int(rating)

def auto_rater_gpt(args, x_0, x_p, tokenizer, model):
    prompt_rater = './autorater/criteria_prompt.txt'

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

    # responses = chatgpt_request(model='gpt-3.5-turbo', message_list=message_list, max_tokens=args.max_length_cot, temperature=args.temperature, sleep=args.api_time_interval)
    # response = responses['choices'][0]['message']['content']
    count = 0
    while count < 5:
        responses = chatgpt_request(model='gpt-3.5-turbo', message_list=message_list, max_tokens=args.max_length_cot, temperature=args.temperature, sleep=args.api_time_interval)
        if not responses:
            return auto_rater(x_0, x_p, args.max_length_cot, tokenizer, model)
        response = responses['choices'][0]['message']['content']
        # print(response)
        
        rating = response[0]

        if rating == '0' or rating == '1':
            print(f'Rating: {rating}')
            return int(rating)
    return auto_rater(x_0, x_p, args.max_length_cot, tokenizer, model)

def auto_rater_question(x_0, x_p, max_len, tokenizer, model):
    prompt_rater = './autorater/criteria_question.txt'

    # Open the file in read mode
    with open(prompt_rater, 'r') as file:
        # Read the entire content of the file
        prompt = file.read().strip()

    # Print the content

    prompt += '\n' + 'Response 0:' + '\n' + x_0 + '\n' + 'Response 1:' + '\n' + x_p + "\n\n" + "Rating: "

    # print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt")
        
    for key in inputs:
        inputs[key] = inputs[key].cuda()
    
    while True:
        with torch.inference_mode():
            generate_ids = model.generate(**inputs, max_new_tokens=max_len, do_sample=False)
        
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        rating = get_rating(response)

        if rating:
            # print(f'Rating: {rating}')
            return int(rating)

def inference_cot_token_attack(args, question_pool, qes_limit, given_prompt):
    correct = 0
    attackable = 0
    wrong_after_attack = 0
    qes_count = 0
    wrong_list = []
    QA_record = []
    success_list = []
    length_lst = []

    model_path = model_libs[args.model]
    tokenizer, model = create_model(model_path, None, 'cuda')

    judge_model_path = model_libs["JudgeLM"]
    tokenizer_judge, model_judge = create_model(judge_model_path, None, 'cuda')

    tokens = tokenizer(' ', return_tensors="pt")
    random_token_id = tokens.input_ids

    for qes_num, qes in enumerate(question_pool):
        if qes_limit is not None and qes_count == qes_limit:
            break
        # create a list for each question to record all answers generated from self-consistency
        all_self_consistency_ans = []
        
        if args.dataset == "last_letters" and args.use_code_style_prompt == True:
            # code style prompt
            prompt = given_prompt + "Q: " + qes['question'] + "\nA: Let's think step by step in Python."
        elif args.basic_cot is True:
            prompt = given_prompt + "Q: " + qes['question'] + "\nA:"
        else:
            prompt = given_prompt + "Q: " + qes['question'] + "\nA: Let's think step by step."

        inputs_in_context = tokenizer(given_prompt + "Q: ", return_tensors="pt")
        for key in inputs_in_context:
            inputs_in_context[key] = inputs_in_context[key].cuda()
        length_in_context = get_embeddings(model, inputs_in_context.input_ids).shape[-2]

        inputs_prompt = tokenizer(prompt, return_tensors="pt")
        for key in inputs_prompt:
            inputs_prompt[key] = inputs_prompt[key].cuda()
        input_attention_mask = inputs_prompt['attention_mask']
        # get the input space embeddings without perturbations
        input_emb = get_embeddings(model, inputs_prompt.input_ids)
        # print(input_emb.shape)
        max_len = args.max_length_cot
        length_query = input_emb.shape[-2]
                
        # enable self-consistency if multipath > 1
        for path in range(0, args.multipath):

            responses = return_response(args, input_emb, input_attention_mask, max_len, tokenizer, model)
            
            QA = {}
            QA['qes_idx'] = qes['question_idx']
            QA['Q'] = qes['question']

            QA['A'] = responses
            QA_record.append(QA)

            pred_ans = answer_extraction(args, responses)

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
            # then we start the attack when we get the correct answer

            # we get the length of the answer
            if args.dataset == 'math':
                a, _ = get_answer_math(responses)
            else:
                a, _ = get_answer(responses)
            # print(a)
            answer = ' ' + a
            input_w_answer = tokenizer(prompt+answer, return_tensors="pt")
            for key in input_w_answer:
                input_w_answer[key] = input_w_answer[key].cuda()
            answer = tokenizer(answer, return_tensors="pt")
            for key in answer:
                answer[key] = answer[key].cuda()
            attention_mask_w_answer = torch.cat((inputs_prompt['attention_mask'], answer['attention_mask']), dim=-1)
            # get the input space embeddings without perturbations
            answer_emb = get_embeddings(model, answer.input_ids)
            # input_emb_w_answer = torch.cat((input_emb, answer_emb), dim=-2)
            input_emb_w_answer = get_embeddings(model, input_w_answer.input_ids)
            length_w_answer = input_emb_w_answer.shape[-2]
            # length_answer = len(tokenizer(value, return_tensors="pt").input_ids[0])

            # we get the length of the cot
            x_0_text = return_cot(responses, a)

            # if auto_rater_gpt_single(args, remove_repeated_phrases(x_0_text), remove_repeated_phrases(x_0_text)) == 0 :
            #     correct -= 1
            #     continue
            # print(x_0_text)
            exp = ' ' + x_0_text
            exp = tokenizer(exp, return_tensors="pt")
            for key in exp:
                exp[key] = exp[key].cuda()
            attention_mask_with_exp = torch.cat((attention_mask_w_answer, exp['attention_mask']), dim=-1)
            # get the input space embeddings without perturbations
            exp_emb = get_embeddings(model, exp.input_ids)
            input_emb_with_exp = torch.cat((input_emb_w_answer, exp_emb), dim=-2)
            length_w_cot = input_emb_with_exp.shape[-2]

            # get whole prompt + response
            # res = tokenizer(' ' + responses, return_tensors="pt")
            # for key in res:
            #     res[key] = res[key].cuda()
            whole = tokenizer(prompt + ' ' + responses, return_tensors="pt")
            for key in whole:
                whole[key] = whole[key].cuda()
            # attention_mask_whole = whole['attention_mask']
            # attention_mask_whole = torch.cat((input_attention_mask, res['attention_mask']), dim=-1)
            # get the input space embeddings without perturbations
            # res_emb = get_embeddings(model, res.input_ids)
            # whole_emb = torch.cat((input_emb, res_emb), dim=-2)
            # whole_emb = get_embeddings(model, whole.input_ids)

            # add perturbation to the prompt
            # perturbed_input_emb= add_radius_noise_to_embedding(input_emb, 0)
            # # get the input space embeddings without perturbations
            # # input_emb_perturb_with_exp = torch.cat((perturbed_input_emb, answer_emb), dim=-2)
            # attention_mask_all_perturb = attention_mask_whole
            
            # # get the input space embeddings without perturbations
            # input_emb_all_perturb = torch.cat((perturbed_input_emb, res_emb), dim=-2)

            # get the true logits
            # outputs_w_answer = model(
            #             # input_ids=input_ids,
            #             attention_mask=attention_mask_whole,
            #             # position_ids=position_ids,
            #             # past_key_values=past_key_values,
            #             inputs_embeds=whole_emb,
            #             # use_cache=False,
            #             output_attentions=False,
            #             output_hidden_states=False,
            #             return_dict=True,
            #             # cache_position=cache_position,
            #         )
            # # print(outputs_w_answer.keys())
            # labels_w_answer = outputs_w_answer['logits']

            # get the attentions
            # outputs_prompt = model(
            #             # input_ids=input_ids,
            #             attention_mask=attention_mask_w_answer,
            #             # position_ids=position_ids,
            #             # past_key_values=past_key_values,
            #             inputs_embeds=input_emb_w_answer,
            #             # use_cache=False,
            #             output_attentions=True,
            #             output_hidden_states=False,
            #             return_dict=True,
            #             # cache_position=cache_position,
            #         )
            # labels_input = outputs_prompt['logits']


            # start attack
            loss = nn.CrossEntropyLoss()
            # topk = args.topk
            # topk = int((length_query - length_in_context) * args.topk_ratio)
            topk_swap = args.topk_swap
            perturb_whole = deepcopy(whole)
            add_tok_num = int((length_query - length_in_context) * args.add_ratio)
            if add_tok_num < 1:
                add_tok_num = 1
            print(f"added token number: {add_tok_num}")
            for step in range(add_tok_num):     
                gc.collect()
                torch.cuda.empty_cache() 

                
                rest = [i+1 for i in range(length_query + step - 2 - length_in_context)]

                # for each input ids, we swap it to another token id in the vacabularies
                count = 0
                idx = 0
                old_q = tokenizer.batch_decode(perturb_whole.input_ids[:, length_in_context-1:length_query-2+step], skip_special_tokens=True)
                while count < topk_swap:
                    # if idx >= len(rest):
                    #     break

                # for idx in rest[:topk_swap]:
                    idx = random.randint(0, len(rest)-1)
                    cur_idx = rest[idx] + length_in_context - 1
                    # old_idx = perturb_whole.input_ids[:, cur_idx]
                    # sample some random tokens
                    # Get vocabulary size
                    vocab_size = tokenizer.vocab_size

                    # Sample a random token ID
                    # random_token_id = torch.tensor(1000).view(-1,1).cuda()
                    random_token_id = torch.randint(0, vocab_size, (1,)).view(-1,1).cuda()
                    # random_token_id = tokenizer.convert_tokens_to_ids('!')
                    # print(random_token_id)
                    # new_idx = top_grad_per_token_answer.indices[cur_idx]
                    # print(perturb_whole.input_ids[:, idx].view(-1, 1).shape)
                    old_token = tokenizer.batch_decode(perturb_whole.input_ids[:, cur_idx].view(-1, 1), skip_special_tokens=True)
                    new_token = tokenizer.batch_decode(torch.tensor(random_token_id).view(-1, 1), skip_special_tokens=True)
                    print(old_token, new_token)
                    # perturb_whole.input_ids[:, cur_idx] = random_token_id]
                    # print(perturb_whole.input_ids.shape)
                    perturb_whole.input_ids = torch.cat((perturb_whole.input_ids[:, :cur_idx], random_token_id.cuda(), perturb_whole.input_ids[:, cur_idx:]), dim=1)
                    new_q = tokenizer.batch_decode(perturb_whole.input_ids[:, length_in_context-1:length_query-2 + 1 + step], skip_special_tokens=True)
                    print(old_q, new_q)
                    # not old_token[0].isdigit() and
                    if auto_rater_question(old_q[0], new_q[0], max_len, tokenizer_judge, model_judge) > 0:
                        count += 1
                        idx += 1
                    else:
                        idx += 1
                        perturb_whole.input_ids[:, cur_idx] = whole.input_ids[:, cur_idx]
                        perturb_whole.input_ids = torch.cat((perturb_whole.input_ids[:, :cur_idx], perturb_whole.input_ids[:, cur_idx+1:]), dim=1)
                    

            whole_emb_perturbed = get_embeddings(model, perturb_whole.input_ids)
            # # cur_prompt = prompt[:length_in_context] + new_q
            prompt = given_prompt + "Q: " + new_q[0] + "A:"
            attention_mask = (whole_emb_perturbed.abs().sum(dim=-1) > 0).int()


            with torch.inference_mode():
                generate_ids = model.generate(inputs_embeds=whole_emb_perturbed[:,:length_query+step+1,:], 
                                attention_mask=attention_mask,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.eos_token_id,
                                max_new_tokens=max_len, do_sample=False)
            # input_query = tokenizer.batch_decode(perturb_whole.input_ids[:, :length_query], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            # print(input_query)
            print('--------------')
            cur_response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(cur_response)

            pred_ans_attack = answer_extraction(args, cur_response)
            x_p_text = return_cot(cur_response, a)
            if pred_ans_attack != qes['answer']:
                wrong_after_attack += 1
                #     break

                # 

                # # if pred_ans_attack == qes['answer'] and (auto_rater(remove_repeated_phrases(x_0_text), remove_repeated_phrases(x_p_text), max_len, tokenizer_judge, model_judge) == 0 or check_all_equations_correct(x_p_text) == 0):
            elif pred_ans_attack == qes['answer'] and (auto_rater_gpt(args, remove_repeated_phrases(x_0_text), remove_repeated_phrases(x_p_text), tokenizer_judge, model_judge) == 0 or check_all_equations_correct(x_p_text) == 0):
                attackable += 1
                success_list.append({'idx':qes['question_idx'], 'new_question': new_q[0].strip().split('\n')[0], 'old_question': qes['question'], 'answer':qes['answer']})
                # break
                # else:
                #     # update input embedding perturb with exp
                #     # input_emb_all_perturb = torch.cat((input_emb_all_perturb[:,:length_query,:], res_emb), dim=-2)
                #     # print('--------------------')
                #     print('===============')
                #     # perturb_whole.input_ids = torch.cat((perturb_whole.input_ids[:, :cur_idx], perturb_whole.input_ids[:, cur_idx+1:]), dim=1)
                #     continue  

            # if step == args.steps - 1:
            else:
                # we continue optimizing the attack by flipping the tokens into some other tokens using token_grads_both
                # get inserted locations by comparing whole.input_ids and whole_perturb.input_ids
                inserted_locations = get_inserted_locations(whole.input_ids, perturb_whole.input_ids)
                print(f"Tokens were inserted at positions: {inserted_locations}")

                original_input_ids = deepcopy(perturb_whole.input_ids)

                # Optional: Verify the insertions
                # for loc in inserted_locations:
                #     original_text = tokenizer.decode(whole.input_ids[0][max(0, loc-1):loc+1])
                #     perturbed_text = tokenizer.decode(perturb_whole.input_ids[0][max(0, loc-1):loc+1])
                #     print(f"At position {loc}:")
                #     print(f"Original context: {original_text}")
                #     print(f"Perturbed context: {perturbed_text}")
                
                # get length_w_answer using cur_response
                if args.dataset == 'math':
                    a, _ = get_answer_math(cur_response)
                else:
                    a, _ = get_answer(cur_response)
                answer = ' ' + a
                input_w_answer = tokenizer(prompt+answer, return_tensors="pt")
                for key in input_w_answer:
                    input_w_answer[key] = input_w_answer[key].cuda()
                answer = tokenizer(answer, return_tensors="pt")
                for key in answer:
                    answer[key] = answer[key].cuda()
                attention_mask_w_answer = torch.cat((inputs_prompt['attention_mask'], answer['attention_mask']), dim=-1)
                # get the input space embeddings without perturbations
                # answer_emb = get_embeddings(model, answer.input_ids)
                # input_emb_w_answer = torch.cat((input_emb, answer_emb), dim=-2)
                input_emb_w_answer = get_embeddings(model, input_w_answer.input_ids)
                length_w_answer = input_emb_w_answer.shape[-2]

                # get labels_w_answer
                outputs_w_answer = model(
                    input_ids=perturb_whole.input_ids,
                    # use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                    # cache_position=cache_position,
                )
                
                labels_w_answer = outputs_w_answer['logits']

                for i in range(args.steps):
                    gc.collect()
                    torch.cuda.empty_cache() 

                    # pring old question
                    old_q = tokenizer.batch_decode(perturb_whole.input_ids[:, length_in_context-1:length_query-1+step], skip_special_tokens=True)

                    # get grads for each token
                    grads_token_both = token_gradients_both(args, model, perturb_whole.input_ids[0], labels_w_answer, loss, length_w_answer)

                    # we consider the token replacement of the inserted locations
                    top_grad_per_token_both = (-grads_token_both).topk(1, dim=1) # select each token with the highest grad possible

                    # print(top_grad_per_token_both.indices)

                    # print(top_grad_per_token_both.values)

                    # select a subset of inserted locations
                    # inserted_locations_sel = random.sample(inserted_locations, int(len(inserted_locations)*0.5))

                    # we select a subset of inserted locations using the gradient information
                    inserted_location_grads = torch.gather(-grads_token_both, 1, torch.tensor(inserted_locations).view(-1, 1).cuda())

                    print(inserted_location_grads.shape)

                    replace_num = int(len(inserted_locations)*args.replace_ratio)
                    if replace_num < 1:
                        replace_num = 1
                    topk_inserted = (inserted_location_grads).topk(replace_num, dim=0)
                    inserted_locations_sel = topk_inserted.indices

                    print(inserted_locations_sel)

                    # we update the perturb_whole.input_ids using the new token with the highest grad
                    for idx in inserted_locations_sel:
                        perturb_whole.input_ids[0][inserted_locations[idx]] = top_grad_per_token_both.indices[inserted_locations[idx]][0]
                        if auto_rater_question(old_q[0], new_q[0], max_len, tokenizer_judge, model_judge) <= 0:
                            perturb_whole.input_ids[0][inserted_locations[idx]] = original_input_ids[0][inserted_locations[idx]]

                    new_q = tokenizer.batch_decode(perturb_whole.input_ids[:, length_in_context-1:length_query-1+step], skip_special_tokens=True)
                    print(old_q, new_q)

                whole_emb_perturbed = get_embeddings(model, perturb_whole.input_ids)
                # cur_prompt = prompt[:length_in_context] + new_q
                attention_mask = (whole_emb_perturbed.abs().sum(dim=-1) > 0).int()


                with torch.inference_mode():
                    generate_ids = model.generate(inputs_embeds=whole_emb_perturbed[:,:length_query+step+1,:], 
                                    attention_mask=attention_mask,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.eos_token_id,
                                    max_new_tokens=max_len, do_sample=False)
                # input_query = tokenizer.batch_decode(perturb_whole.input_ids[:, :length_query], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                # print(input_query)
                print('--------------')
                cur_response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                print(cur_response)

                pred_ans_attack = answer_extraction(args, cur_response)
                if pred_ans_attack != qes['answer']:
                    wrong_after_attack += 1
                    # break

                x_p_text = return_cot(cur_response, a)

                # if pred_ans_attack == qes['answer'] and (auto_rater(remove_repeated_phrases(x_0_text), remove_repeated_phrases(x_p_text), max_len, tokenizer_judge, model_judge) == 0 or check_all_equations_correct(x_p_text) == 0):
                if pred_ans_attack == qes['answer'] and (auto_rater_gpt(args, remove_repeated_phrases(x_0_text), remove_repeated_phrases(x_p_text), tokenizer_judge, model_judge) == 0 or check_all_equations_correct(x_p_text) == 0):
                    attackable += 1
                    success_list.append({'idx':qes['question_idx'], 'new_question': new_q[0].strip().split('\n')[0], 'old_question': qes['question'], 'answer':qes['answer']})
                    # break
                # else:
                #     # update input embedding perturb with exp
                #     # input_emb_all_perturb = torch.cat((input_emb_all_perturb[:,:length_query,:], res_emb), dim=-2)
                #     # print('--------------------')
                #     print('===============')
                #     # perturb_whole.input_ids = torch.cat((perturb_whole.input_ids[:, :cur_idx], perturb_whole.input_ids[:, cur_idx+1:]), dim=1)
                #     continue
                            
            length_lst.append(step)       
        else:
            wrong_list.append({'idx':qes['question_idx'], 'pred_ans':final_consistent_ans, 'GT':qes['answer']})

        qes_count += 1
    print(length_lst)
    avg_length = sum(length_lst) / len(length_lst)

    return correct, attackable, wrong_after_attack, avg_length, wrong_list, success_list, QA_record


def arg_parser():
    parser = argparse.ArgumentParser(description="CoT")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["math", "gsm8k","svamp", "aqua", "csqa", "asdiv", "last_letters", "addsub", "singleeq", "strategyqa", "multiarith", "time_zone"], help="dataset to inference"
    )
    parser.add_argument(
        "--prompt_path", type=str, default="./inference_prompts/gsm8k_k=10", help="prompts to use"
    )
    parser.add_argument(
        "--model", type=str, default="llama3-8B", choices=["llama3-8B","mistral", "zephyr", "qwen", "deepseek"], help="model used for decoding."
    )
    parser.add_argument(
        "--method", type=str, default="active_cot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "auto_cot", "active_cot"], help="method"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results/token_attack", help="output directory"
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
        "--temperature", type=float, default=0.7, help=""
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
    parser.add_argument(
        "--steps", type=int, default=5, help="num of steps for the attack"       
    )
    parser.add_argument(
        "--replace_ratio", type=float, default=0.25, help="num of tokens looking for overlapping"       
    )
    parser.add_argument(
        "--add_ratio", type=float, default=0.2, help="num of tokens for swapping each step"       
    )
    parser.add_argument(
        "--topk_swap", type=int, default=1, help="num of tokens for swapping each step"       
    )
    parser.add_argument(
        "--gpt_judge", default=False, action='store_true'
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
        args.dataset_path = "./dataset/GSM8K/test.jsonl"
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
        args.dataset_path = "./dataset/strategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters_test.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/MAWPS/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/MAWPS/SingleEq.json"
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
