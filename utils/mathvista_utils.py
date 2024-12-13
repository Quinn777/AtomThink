import pandas as pd
from Levenshtein import distance
import os
import re
from tqdm import tqdm
from openai import OpenAI
from prompts import demo_prompt
from mathverse_utils import read_json, save_json
import json
import time
from loguru import logger


def get_most_similar(prediction, choices):
    """
    Use the Levenshtein distance (or edit distance) to determine which of the choices is most similar to the given prediction
    """
    distances = [distance(prediction, choice) for choice in choices]
    ind = distances.index(min(distances))
    return choices[ind]


def normalize_extracted_answer(
        extraction, choices, question_type, answer_type, precision, ignore_empty_extractions=False
):
    """
    Normalize the extracted answer to match the answer type
    """
    if question_type == 'multi_choice':
        # make sure the extraction is a string
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            try:
                extraction = str(extraction)
            except Exception:
                extraction = ""
        if ignore_empty_extractions and not extraction:
            return None
        letter = re.findall(r'\(([a-zA-Z])\)', extraction)
        if len(letter) > 0:
            extraction = letter[0].upper()
        sequential_characters = [chr(ord('A') + i) for i in range(len(choices))]
        if extraction in sequential_characters:
            option_index = sequential_characters.index(extraction)
            normalized_extraction = choices[option_index]
        else:
            normalized_extraction = get_most_similar(extraction, choices)
        assert normalized_extraction in choices
    elif answer_type == 'integer':
        try:
            normalized_extraction = str(int(float(extraction)))
        except Exception:
            normalized_extraction = None
    elif answer_type == 'float':
        try:
            normalized_extraction = str(round(float(extraction), precision))
        except Exception:
            normalized_extraction = None
    elif answer_type == 'list':
        try:
            normalized_extraction = str(extraction)
        except Exception:
            normalized_extraction = None

    return normalized_extraction


def safe_equal(prediction, answer):
    """
    Check if the prediction is equal to the answer, even if they are of different types
    """
    try:
        if prediction == answer:
            return True
        return False
    except Exception as e:
        logger.info(e)
        return False


def get_acc_with_contion(res_pd, key, value):
    if key == 'skills':
        total_pd = res_pd[res_pd[key].apply(lambda x: value in x)]
    else:
        total_pd = res_pd[res_pd[key] == value]
    correct_pd = total_pd[total_pd['true_false'] == True]
    acc = len(correct_pd) / len(total_pd)
    return len(correct_pd), len(total_pd), acc


def calculate_score_func(answers_file, max_num_problems, ignore_empty_extractions, caculate_gain, random_file,
                         scores_file_path):
    logger.info("MathVista: Calculating Scores - Start")
    output_file_path = answers_file
    ground_truth_problems = read_json(output_file_path)
    logger.info(f"Reading {output_file_path}...")
    results = read_json(output_file_path)
    full_pids = list(results.keys())
    max_num_problems = min(len(full_pids), max_num_problems)
    logger.info(f"Number of test problems to run: {max_num_problems}")
    test_pids = full_pids[:max_num_problems]
    logger.info("For each problem normalize extractions and get True False value")
    update_json_flag = False
    for i, pid in enumerate(tqdm(test_pids)):
        problem = results[pid]
        choices = problem['choices']
        question_type = problem['question_type']
        answer_type = problem['answer_type']
        precision = problem['precision']
        extraction = problem['extraction']
        if 'answer' in problem:
            answer = problem['answer']
        else:
            answer = ground_truth_problems[i]['answer']
            problem['answer'] = answer
        prediction = normalize_extracted_answer(
            extraction,
            choices,
            question_type,
            answer_type,
            precision,
            ignore_empty_extractions=ignore_empty_extractions,
        )
        true_false = safe_equal(prediction, answer)

        if "true_false" not in problem:
            update_json_flag = True
        elif true_false != problem['true_false']:
            update_json_flag = True
        if "prediction" not in problem:
            update_json_flag = True
        elif prediction != problem['prediction']:
            update_json_flag = True
        problem['prediction'] = prediction
        problem['true_false'] = true_false
    if update_json_flag:
        logger.info("Updating input file with predictions and true_false...")
        save_json(output_file_path, results)
        logger.info(f"Saved {output_file_path}")
    logger.info("Calculate the average accuracy")
    total = len(results)
    correct = 0
    for pid in tqdm(test_pids):
        if results[pid]['true_false']:
            correct += 1
    accuracy = correct / total
    scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}
    for pid in tqdm(test_pids):
        results[pid].update(results[pid].pop('metadata'))
    results_df = pd.DataFrame(results).T

    target_keys = [
        'question_type',
        'answer_type',
        'language',
        'source',
        'category',
        'task',
        'context',
        'grade',
        'skills',
    ]

    for key in target_keys:
        # get the unique values of the key
        if key == 'skills':
            values = []
            for i in range(len(results_df)):
                values += results_df[key][i]
            values = list(set(values))
        else:
            values = results_df[key].unique()
        scores[key] = {}
        for value in values:
            correct, total, acc = get_acc_with_contion(results_df, key, value)
            if total > 0:
                scores[key][value] = {"accuracy": acc, "correct": correct, "total": total}
        scores[key] = dict(sorted(scores[key].items(), key=lambda item: float(item[1]['accuracy']), reverse=True))
    output_dir = os.path.dirname(answers_file)
    if caculate_gain:
        random_file = os.path.join(output_dir, random_file)
        random_scores = json.load(open(random_file))
        logger.info("Calculating the score gains...")
        for key in scores:
            if key == 'average':
                gain = round(float(scores[key]['accuracy']) - float(random_scores[key]['accuracy']), 2)
                scores[key]['acc_gain'] = gain
            else:
                for sub_key in scores[key]:
                    gain = round(
                        float(scores[key][sub_key]['accuracy']) - float(random_scores[key][sub_key]['accuracy']), 2
                    )
                    scores[key][sub_key]['acc_gain'] = str(gain)

    metrics_str = get_full_metrics_str(scores)
    logger.info(metrics_str)
    with open(scores_file_path, 'w') as f:
        json.dump(scores, f, indent=4)
    logger.info(f"Saved scores to: {scores_file_path}")
    logger.info("MathVista: Calculating Scores - Finish")


def get_full_metrics_str(metrics_dict) -> str:
    divider = "=" * 40
    avg_accuracy = metrics_dict["average"]["accuracy"]
    avg_correct = metrics_dict["average"]["correct"]
    avg_total = metrics_dict["average"]["total"]

    metrics_str = f"""
{f"Correct: {avg_correct}/{avg_total} - Accuracy: {avg_accuracy * 100:.2f}%"}
{divider}
""".lstrip()

    for key, item in metrics_dict.items():
        if key == "average":
            continue
        formatted_item_dict = {}
        for sub_key, sub_item in item.items():
            acc = sub_item["accuracy"]
            correct = sub_item["correct"]
            total = sub_item["total"]
            values = [f"{acc * 100:.2f}%", f"({correct}/{total})"]
            formatted_item_dict[sub_key] = values
        category_df = pd.DataFrame(formatted_item_dict, index=["Accuracy", "Correct/Total"])
        metrics_str += f"""
{key}
{divider}
{category_df.T}
"""
    return metrics_str


def make_api_call(messages, max_tokens, temperature=0.2, is_json=False, custom_client=None, model='gpt-4o'):
    if custom_client != None:
        client = custom_client
    else:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", None)
        )
        client.base_url = 'http://rerverseapi.workergpt.cn/v1'
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


def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


def extract_answer(response, problem, quick_extract=False):
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    query = problem['query']
    pid = problem['pid']
    if response == "":
        return ""
    if question_type == 'multi_choice' and response in choices:
        return response
    if answer_type == "integer":
        try:
            extraction = int(response)
            return str(extraction)
        except Exception as e:
            pass
    if answer_type == "float":
        try:
            extraction = str(float(response))
            return extraction
        except Exception as e:
            pass

    if quick_extract:
        logger.info("Quickly extracting answer...")
        try:
            result = re.search(r'The answer is "(.*)"\.', response)
            if result:
                extraction = result.group(1)
                return extraction
        except Exception as e:
            pass

    try:
        full_prompt = create_test_prompt(demo_prompt, query, response)
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
        extraction = make_api_call(msg, 500)
        return extraction
    except Exception as e:
        logger.info(f"Error in extracting answer for problem: {pid} with response: {response}")
        logger.info(e)

    return ""


def gpt4o(messages, temperature=0.9, model="gpt-4o"):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", None)
    )
    client.base_url = 'http://rerverseapi.workergpt.cn/v1'
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature
    )
    content = chat_completion.choices[0].message.content
    # print("content", content)
    return content