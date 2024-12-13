from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import torch
import re
from collections import Counter


def construct_prompt_from_rollout(prompt, rollout):
    steps = "\n".join(rollout)
    prompt = prompt.replace("HISTORICAL REASONING STEPS:\n", "HISTORICAL REASONING STEPS:\n"+steps)
    return prompt


def add_step_into_hist(prompt, response):
    seg_str = "\n\nYour task is to predict the next step of reasoning"
    prompt_list = prompt.split(seg_str)
    prompt_list[0] = prompt_list[0] + response
    prompt = seg_str.join(prompt_list)
    return prompt


def inference_one_step(init_prompt, model, img_list, image_size, tokenizer, args):
    input_ids = tokenizer_image_token(init_prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                      return_tensors='pt').unsqueeze(0).to("cuda:0")
    pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    attention_masks = input_ids.ne(pad_token_ids).to("cuda:0")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images=img_list,
            image_sizes=image_size,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            max_new_tokens=1024,
            pad_token_id=pad_token_ids,
            use_cache=True)
    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response

def txt_verifier(response1, response2):
    # check null
    if not check_response_is_repeat(response1):
        return False, False
    # check repeat
    if get_repeated_sentence_ratio(response1) > 0.45:
        return False, False
    if jaccard_similarity(response1, response2) > 0.98:
        return False, False
    answer_str = ["final answer is", "final answer:", "Final Answer:", "Final Answer is", "Final answer:",
                  "Final answer is"]
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

