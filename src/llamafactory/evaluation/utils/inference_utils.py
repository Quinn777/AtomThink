import torch
import re
from collections import Counter


def update_inputs_from_rollout(ori_inputs, rollout, processor):
    steps = "\n\n".join(rollout)
    prompt = ori_inputs['prompt'].replace("HISTORICAL REASONING STEPS:\n", "HISTORICAL REASONING STEPS:\n" + steps)
    ori_inputs['inputs'] = processor(
        text=[prompt],
        images=ori_inputs['images'],
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    ori_inputs['prompt'] = prompt
    return ori_inputs


def add_step_into_hist(processor, inputs, response):
    prompt = inputs['prompt']
    seg_str = "\n\nYour task is to predict the next step of reasoning"
    seg_str2 = "\n\nPlease think through the outline first and then proceed with reasoning step by step."
    if seg_str in prompt:
        prompt_list = prompt.split(seg_str)
        if prompt_list[0].endswith('HISTORICAL REASONING STEPS:\n'):
            prompt_list[0] = prompt_list[0] + response
        else:
            prompt_list[0] = prompt_list[0] + '\n\n' + response
        prompt = seg_str.join(prompt_list)
    else:
        prompt_list = prompt.split(seg_str2)
        if prompt_list[0].endswith('HISTORICAL REASONING STEPS:\n'):
            prompt_list[0] = prompt_list[0] + response
        else:
            prompt_list[0] = prompt_list[0] + '\n\n' + response
        prompt = seg_str2.join(prompt_list)
    inputs['inputs'] = processor(
        text=[prompt],
        images=inputs['images'],
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    inputs['prompt'] = prompt
    return inputs


def inference_one_step(inputs, model, processor, generating_args, temperature):
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=generating_args.top_p,
            top_k=generating_args.top_k,
            max_new_tokens=generating_args.max_new_tokens,
            use_cache=True)
    output_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return output_text


def txt_verifier(response1, response2):
    # check null
    if not check_response_is_repeat(response1):
        return False, False
    # check repeat
    if get_repeated_sentence_ratio(response1) > 0.45:
        return False, False
    if jaccard_similarity(response1, response2) > 0.98:
        return False, False
    # length control
    if len(response1.split(' ')) > 500:
        return False, False
    answer_str = ["final answer is", "final answer:", "Final Answer:", "Final Answer is", "Final answer:",
                  "Final answer is", "The answer is", "the answer is"]
    for s in answer_str:
        if s in response1:
            return True, True
    return True, False


def check_response_is_repeat(response):
    if not response:
        return False
    if response.count("Step") > 2 or response.count("step") > 2:
        return False
    return True


def jaccard_similarity(str1, str2):
    str1 = re.sub(r"Step \d+: ", "", str1)
    str2 = re.sub(r"Step \d+: ", "", str2)
    set1 = set(str1)
    set2 = set(str2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if union == 0:
        return 1.0
    similarity = intersection / union
    return similarity


def get_repeated_sentence_ratio(text):
    sentences = re.split(r'[.!?]', text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    sentence_counts = Counter(sentences)
    total_sentences = len(sentences)
    repeated_sentences = sum(1 for count in sentence_counts.values() if count > 1)
    if total_sentences == 0:
        return 0
    repeated_ratio = repeated_sentences / total_sentences
    return repeated_ratio
