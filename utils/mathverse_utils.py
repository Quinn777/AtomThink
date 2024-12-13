import json
import time
import pickle
import re
import os
import copy
import argparse
from tqdm import tqdm
from collections import defaultdict
from prompts import demo_prompt_extract, demo_prompt_score
from openai import OpenAI
from loguru import logger
import PIL.Image as Image


def create_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def read_csv(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            data.append(line.strip())
    return data


def read_pandas_csv(csv_path):
    # read a pandas csv sheet
    import pandas as pd
    df = pd.read_csv(csv_path)
    return df


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_jsonl(file):
    with open(file, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def contains_digit(text):
    if any(char.isdigit() for char in text):
        return True
    return False


def contains_quantity_word(text, special_keep_words=[]):
    quantity_words = ["most", "least", "fewest"
                                       "more", "less", "fewer",
                      "largest", "smallest", "greatest",
                      "larger", "smaller", "greater",
                      "highest", "lowest", "higher", "lower",
                      "increase", "decrease",
                      "minimum", "maximum", "max", "min",
                      "mean", "average", "median",
                      "total", "sum", "add", "subtract",
                      "difference", "quotient", "gap",
                      "half", "double", "twice", "triple",
                      "square", "cube", "root",
                      "approximate", "approximation",
                      "triangle", "rectangle", "circle", "square", "cube", "sphere", "cylinder", "cone", "pyramid",
                      "multiply", "divide",
                      "percentage", "percent", "ratio", "proportion", "fraction", "rate",
                      ]
    quantity_words += special_keep_words  # dataset specific words
    words = re.findall(r'\b\w+\b', text)  # This regex pattern matches any word in the text
    if any(word in quantity_words for word in words):
        return True
    return False  # If none of the words could be converted to a number, return False


def is_bool_word(text):
    if text in ["Yes", "No", "True", "False",
                "yes", "no", "true", "false",
                "YES", "NO", "TRUE", "FALSE"]:
        return True
    return False


def is_digit_string(text):
    # remove ".0000"
    text = text.strip()
    text = re.sub(r'\.0+$', '', text)
    try:
        int(text)
        return True
    except ValueError:
        return False


def is_float_string(text):
    if "." in text:
        try:
            float(text)
            return True
        except ValueError:
            return False
    return False


def copy_image(image_path, output_image_path):
    from shutil import copyfile
    copyfile(image_path, output_image_path)


def copy_dir(src_dir, dst_dir):
    from shutil import copytree
    copytree(src_dir, dst_dir)


def get_image_size(img_path):
    img = Image.open(img_path)
    width, height = img.size
    return width, height


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


def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


def create_test_prompt1(demo_prompt, response, inst):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"Model response: '{response}'\nExtracted Answer: "
    full_prompt = f"{demo_prompt}\n\n{test_prompt}"
    return full_prompt


def extract_answer(response, inst, api_key):
    # general extraction
    try:
        full_prompt = create_test_prompt1(demo_prompt_extract, response, inst)
        extraction = get_chat_response(full_prompt, api_key)
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


def match_answer(inst, api_key, quick_match=False):
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


def extract_answer_s1(model_output_file, answer_extraction_file, trunk, api_key, cache):
    result_file = model_output_file
    print(f"Reading {result_file}...")
    results = read_json(result_file)
    os.makedirs(os.path.dirname(answer_extraction_file), exist_ok=True)
    if os.path.exists(answer_extraction_file):
        save_results = json.load(open(answer_extraction_file))
    else:
        save_results = []
    for i, inst in enumerate(tqdm(results)):
        save_inst = copy.deepcopy(inst)
        if cache:
            pass
        else:
            if 'response' in save_inst:
                response = save_inst['response']
            else:
                response = ''
                print(save_inst)
                print("######### NO MODEL ANSWER ###########")  # some model may output nothing due to safety
            response = trunk_response(response, trunk)
            extraction = extract_answer(response, save_inst, api_key)
            save_inst['extraction'] = extraction.replace('Extracted Answer: ', '').strip()  # sometimes gpt will repeat
            save_results.append(save_inst)
    print(f"Saving results to {answer_extraction_file}...")
    save_json(save_results, answer_extraction_file)


def score_answer_s2(answer_extraction_file, save_file, cache, api_key, quick_match):
    result_file = answer_extraction_file
    print(f"Reading {result_file}...")
    results = read_json(result_file)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    save_results = []
    score_dict = defaultdict(lambda: defaultdict(list))
    score_version_dict = defaultdict(list)
    for i, inst in enumerate(tqdm(results)):
        save_inst = copy.deepcopy(inst)
        if cache and 'judgement' in save_inst:
            pass
        else:
            judgement = match_answer(save_inst, api_key, quick_match)
            try_n = 10
            while try_n > 0:
                if judgement.strip() not in ['0', '1']:
                    print('Wrong return format: ', judgement)
                    judgement = match_answer(save_inst, api_key, quick_match)
                    try_n -= 1
                    save_inst['judgement'] = 0
                else:
                    save_inst['judgement'] = int(judgement)
                    break

            save_results.append(save_inst)
        score_dict[save_inst['metadata']['subject']][save_inst['metadata']['subfield']].append(save_inst['judgement'])
        score_version_dict[save_inst['problem_version']].append(save_inst['judgement'])
    print(f"Saving results to {save_file}...")
    save_json(save_results, save_file)
    logger.info(f"Results saved.")

    total_cnt, right_cnt = 0, 0
    for subject in score_dict:
        subject_total_cnt, subject_right_cnt = 0, 0
        for subfield in score_dict[subject]:
            subfield_total_cnt = len(score_dict[subject][subfield])
            subfield_right_cnt = len([inst for inst in score_dict[subject][subfield] if inst == 1])
            subject_total_cnt += subfield_total_cnt
            subject_right_cnt += subfield_right_cnt
            logger.info(f"{subject}-{subfield} Acc: {(subfield_right_cnt / subfield_total_cnt):.3f}")
        logger.info(f"{subject} Acc: {(subject_right_cnt / subject_total_cnt):.3f}")
        total_cnt += subject_total_cnt
        right_cnt += subject_right_cnt
    logger.info(f"Total Acc: {(right_cnt / total_cnt):.3f}")
    total_cnt, right_cnt = 0, 0
    for version in score_version_dict:
        version_total_cnt = len(score_version_dict[version])
        version_right_cnt = len([inst for inst in score_version_dict[version] if inst == 1])
        total_cnt += version_total_cnt
        right_cnt += version_right_cnt
        logger.info(f"{version} Acc: {(version_right_cnt / version_total_cnt):.3f}")
        logger.info(version_total_cnt)

    logger.info(f"Acc: {(right_cnt / total_cnt):.3f}")


def show_scores(score_file):
    results = read_json(score_file)
    score_dict = defaultdict(lambda: defaultdict(list))
    score_version_dict = defaultdict(list)
    for i, inst in enumerate(tqdm(results)):
        save_inst = copy.deepcopy(inst)
        score_dict[save_inst['metadata']['subject']][save_inst['metadata']['subfield']].append(save_inst['judgement'])
        score_version_dict[save_inst['problem_version']].append(save_inst['judgement'])
    total_cnt, right_cnt = 0, 0
    for subject in score_dict:
        subject_total_cnt, subject_right_cnt = 0, 0
        for subfield in score_dict[subject]:
            subfield_total_cnt = len(score_dict[subject][subfield])
            subfield_right_cnt = len([inst for inst in score_dict[subject][subfield] if inst == 1])
            subject_total_cnt += subfield_total_cnt
            subject_right_cnt += subfield_right_cnt
            logger.info(f"{subject}-{subfield} Acc: {(subfield_right_cnt / subfield_total_cnt):.3f}")
        logger.info(f"{subject} Acc: {(subject_right_cnt / subject_total_cnt):.3f}")
        total_cnt += subject_total_cnt
        right_cnt += subject_right_cnt
    logger.info(f"Total Acc: {(right_cnt / total_cnt):.3f}")
    total_cnt, right_cnt = 0, 0
    for version in score_version_dict:
        version_total_cnt = len(score_version_dict[version])
        version_right_cnt = len([inst for inst in score_version_dict[version] if inst == 1])
        total_cnt += version_total_cnt
        right_cnt += version_right_cnt
        logger.info(f"{version} Acc: {(version_right_cnt / version_total_cnt):.3f}")
        logger.info(version_total_cnt)

    logger.info(f"Acc: {(right_cnt / total_cnt):.3f}")

