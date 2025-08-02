import json
import os
from tqdm import tqdm
import time
from llamafactory.evaluation.utils.prompts import demo_prompt_score
from openai import OpenAI
from loguru import logger
import re
from os.path import join, isdir, isfile, isdir, dirname
import yaml
import time

MAX_TRY = 5

def s0_merge_answers(model_output_dir, dataset_name, tag):
    inference_dir = join(model_output_dir, 'inference', dataset_name)
    answers_file = join(inference_dir, f'answers_{tag}.json')

    subfolders = []
    for entry in os.listdir(inference_dir):
        entry_path = os.path.join(inference_dir, entry)
        if os.path.isdir(entry_path) and 'tmp' not in entry_path:
            subfolders.append(entry_path)
    ans_list = []
    ans_dict = {}
    for subfolder in subfolders:
        for filename in os.listdir(subfolder):
            file = os.path.join(subfolder, filename)
            if os.path.isfile(file) and tag in filename:
                answers = read_json(file)
                if isinstance(answers, dict):
                    ans_dict = {**ans_dict, **answers}
                else:
                    ans_list.extend(answers)
    if ans_dict:
        save_json(answers_file, ans_dict)
        print(f'Total Answers Length is: {len(ans_dict)}')
        time.sleep(5)
    else:
        save_json(answers_file, ans_list)
        print(f'Total Answers Length is: {len(ans_list)}')
        time.sleep(5)
    return answers_file

def s1_separate_n(answers_file, n_process):
    split_dir = dirname(answers_file)
    split_dir = join(split_dir, 'tmp')
    if not isdir(split_dir):
        os.makedirs(split_dir)
    total_answers = read_json(answers_file)
    if isinstance(total_answers, dict):
        total_answers = list(total_answers.items())
        output_type = "dict"
    else:
        output_type = "list"
    answers_file_list = []
    total_length = len(total_answers)
    chunk_size = total_length // n_process
    remainder = total_length % n_process

    start = 0
    for i in range(n_process):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunk = total_answers[start:end]

        if output_type == "dict":
            chunk = dict(chunk)
        filename = f"{i}.json"
        filename = join(split_dir, filename)
        save_json(filename, chunk)
        answers_file_list.append(filename)
        print(f"Saved {filename} with {len(chunk)} elements.")
        start = end
    return answers_file_list

def s3_merge_save_answers(answers_file, answers_file_list):
    ans_list = []
    ans_dict = {}
    for file in answers_file_list:
        try:
            if '.jsonl' in file:
                answers = read_json(file)
            else:
                answers = read_json(file)
        except Exception as e:
            print(f'error: {e}\n{file}')
            continue
        if isinstance(answers, dict):
            ans_dict = {**ans_dict, **answers}
        else:
            ans_list.extend(answers)
    if isinstance(answers, dict):
        save_json(answers_file, ans_dict)
    else:
        save_json(answers_file, ans_list)

def save_jsonl(path: str, data: list, ) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for line in tqdm(data, desc='save'):
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

def save_json(filename, ds):
    with open(filename, 'w') as f:
        json.dump(ds, f, indent=4)

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def read_jsonl(path: str, key: str = None):
    data = []
    with open(os.path.expanduser(path)) as f:
        for line in f:
            if not line:
                continue
            data.append(json.loads(line))

    if key is not None:
        data.sort(key=lambda x: x[key])
        data = {item[key]: item for item in data}
    return data

def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return yaml_dict

def save_yaml(data, save_path):
    with open(save_path, "w") as f:
        yaml.dump(data, f)

def score_func(response, query, gt):
    if not response:
        return 0
    try:
        full_prompt = demo_prompt_score.strip().format(question=query, extraction=response, gt=gt)
        msg = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": full_prompt
                    },
                ]
            }
        ]
        try_n = 0
        while try_n < MAX_TRY:
            score = make_api_call(msg, 50)
            if 'Judgement: ' in score:
                score = score.split('Judgement: ')[-1]
            elif 'Judgement:' in score:
                score = score.split('Judgement:')[-1]
            elif 'judgement: ' in score:
                score = score.split('judgement: ')[-1]
            elif 'judgement:' in score:
                score = score.split('judgement:')[-1]
            try:
                if int(score) == 0 or int(score) == 1:
                    return int(score)
            except Exception as e:
                logger.warning(f"Error in extracting answer!\nProblem:{query}\nResponse:{response}\nGT:{gt}\nGPT response: {score}")
                continue

    except Exception as e:
        logger.warning(f"score_func Error! problem: {query} with response: {response}")
        logger.warning(e)
        return None

def make_api_call(messages, max_tokens=200, temperature=0.2, is_json=False, custom_client=None, model='gpt-4o'):
    if custom_client != None:
        client = custom_client
    else:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", None)
        )
        client.base_url = ''
    for attempt in range(5):
        try:
            if not is_json:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt == 4:
                return f"Error: {str(e)}"
            time.sleep(1)  # Wait for 1 second before retrying

def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt

def extract_by_rule(response):
    try:
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    except Exception as e:
        pass
    try:
        pattern = r"the final answer is: (.+?)\."
        match = re.search(pattern, response)
        if match:
            return match.group(1)
        pattern = r"The answer is: (.+?)\."
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    except Exception as e:
        pass
    try:
        response = int(response)
    except Exception as e:
        pass
    try:
        response = str(float(response))
    except Exception as e:
        pass
    return response