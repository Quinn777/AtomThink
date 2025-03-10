import time
import pickle
import re
import os
import copy
from tqdm import tqdm
from collections import defaultdict
from llamafactory.evaluation.utils.prompts import demo_prompt_extract, demo_prompt_score
from llamafactory.evaluation.utils.eval_utils import read_json, save_json, save_jsonl, read_jsonl, create_test_prompt, make_api_call, score_func, extract_by_rule
from openai import OpenAI
from loguru import logger
import PIL.Image as Image

def s2_extract_and_score(chunk_file):
    result_file = chunk_file
    print(f"Reading {result_file}...")
    results = read_json(result_file)

    save_results = []
    for i, inst in enumerate(tqdm(results)):
        save_inst = copy.deepcopy(inst)
        if 'response' in save_inst:
            response = save_inst['response']
        else:
            response = ''
            print(save_inst)
            print("######### NO MODEL ANSWER ###########")  # some model may output nothing due to safety
        if ('extraction' not in save_inst) or ('score' not in save_inst):
            extraction = extract_by_rule(response)
            score = score_func(extraction, save_inst['question'], save_inst['answer'])
            save_inst['extraction'] = extraction
            save_inst['score'] = score
        elif 'score' in save_inst:
            if save_inst['score'] == None:
                extraction = extract_by_rule(response)
                score = score_func(extraction, save_inst['question'], save_inst['answer'])
                save_inst['extraction'] = extraction
                save_inst['score'] = score
        save_results.append(save_inst)
    print(f"Saving results to {chunk_file}...")
    save_json(chunk_file, save_results)


def s4_show_scores(answers_file):
    result_file = answers_file
    print(f"Reading {result_file}...")
    results = read_json(result_file)
    score_file = answers_file.split(".json")[0] + '_result.txt'

    with open(score_file, "w") as file:
        score_dict = defaultdict(lambda: defaultdict(list))
        score_version_dict = defaultdict(list)
        for i, inst in enumerate(tqdm(results)):
            save_inst = copy.deepcopy(inst)
            score_dict[save_inst['metadata']['subject']][save_inst['metadata']['subfield']].append(save_inst['score'])
            score_version_dict[save_inst['problem_version']].append(save_inst['score'])
        total_cnt, right_cnt = 0, 0
        for subject in score_dict:
            subject_total_cnt, subject_right_cnt = 0, 0
            for subfield in score_dict[subject]:
                subfield_total_cnt = len(score_dict[subject][subfield])
                subfield_right_cnt = len([inst for inst in score_dict[subject][subfield] if inst == 1])
                subject_total_cnt += subfield_total_cnt
                subject_right_cnt += subfield_right_cnt
                logger.info(f"{subject}-{subfield} Acc: {(subfield_right_cnt / subfield_total_cnt):.3f}")
                file.write(f"{subject}-{subfield} Acc: {(subfield_right_cnt / subfield_total_cnt):.3f}")
            logger.info(f"{subject} Acc: {(subject_right_cnt / subject_total_cnt):.3f}")
            file.write(f"{subject} Acc: {(subject_right_cnt / subject_total_cnt):.3f}")
            total_cnt += subject_total_cnt
            right_cnt += subject_right_cnt
        logger.info(f"Total Acc: {(right_cnt / total_cnt):.3f}")
        file.write(f"Total Acc: {(right_cnt / total_cnt):.3f}")

        total_cnt, right_cnt = 0, 0
        for version in score_version_dict:
            version_total_cnt = len(score_version_dict[version])
            version_right_cnt = len([inst for inst in score_version_dict[version] if inst == 1])
            total_cnt += version_total_cnt
            right_cnt += version_right_cnt
            logger.info(f"{version} Acc: {(version_right_cnt / version_total_cnt):.3f}")
            logger.info(version_total_cnt)
            file.write(f"{version} Acc: {(version_right_cnt / version_total_cnt):.3f}")
        logger.info(f"Acc: {(right_cnt / total_cnt):.3f}")
        file.write(f"Acc: {(right_cnt / total_cnt):.3f}")

def create_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
def read_csv(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            data.append(line.strip())
    return data

def get_chat_response(prompt, api_key=None, model="gpt-4o", temperature=0.2, max_tokens=256, n=1, patience=10,
                      sleep_time=0):
    messages = [
        {"role": "user", "content": prompt},
    ]
    while patience > 0:
        patience -= 1
        try:
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", None)
            )
            client.base_url = 'http://rerverseapi.workergpt.cn/v1'
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                n=n,
                temperature=temperature,
            )
            if n == 1:
                prediction = response.choices[0].message.content.strip()
                if prediction != "" and prediction != None:
                    return prediction
            else:
                prediction = [choice['message']['content'].strip() for choice in response['choices']]
                if prediction[0] != "" and prediction[0] != None:
                    return prediction

        except Exception as e:
            if "Rate limit" not in str(e):
                print(e)
            if "Please reduce the length of the messages" in str(e):
                print("!!Reduce promot size")
                new_size = int(len(prompt) * 0.9)
                new_start = len(prompt) - new_size
                prompt = prompt[new_start:]
                messages = [
                    {"role": "user", "content": prompt},
                ]
            if sleep_time > 0:
                time.sleep(sleep_time)
    return ""

def create_test_prompt1(demo_prompt, response, inst):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"Model response: '{response}'\nExtracted Answer: "
    full_prompt = f"{demo_prompt}\n\n{test_prompt}"
    return full_prompt

def extract_answer(response, inst):
    # general extraction
    try:
        full_prompt = create_test_prompt1(demo_prompt_extract, response, inst)
        extraction = get_chat_response(full_prompt,)
        return extraction
    except Exception as e:
        print(e)
        print(f"Error in extracting answer for {response}")
    return ""

def trunk_response(response, trunk_length):
    if trunk_length <= 0:
        return response
    else:
        return_res = ' '.join(response.split(' ')[-trunk_length:])
        return return_res

def create_test_prompt2(demo_prompt, inst):
    demo_prompt = demo_prompt.strip()
    full_prompt = demo_prompt.format(question=inst['question'], gt=inst['answer'], extraction=inst['extraction'])
    return full_prompt

def match_answer(inst, api_key=None, quick_match=False):
    # quick match
    if quick_match:
        return '1' if inst['answer'] == inst['extraction'] else '0'
    try:
        full_prompt = create_test_prompt2(demo_prompt_score, inst)
        extraction = get_chat_response(full_prompt, api_key)
        return extraction.replace("Judgement:", "").strip()
    except Exception as e:
        print(e)
        print(f"Error in matching answer")

    return ""