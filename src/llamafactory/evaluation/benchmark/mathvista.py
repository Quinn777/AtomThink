import pandas as pd
from Levenshtein import distance
import os
import re
from tqdm import tqdm
from openai import OpenAI
from llamafactory.evaluation.utils.prompts import demo_prompt
from llamafactory.evaluation.utils.eval_utils import read_json, save_json, save_jsonl, read_jsonl, create_test_prompt, make_api_call, score_func
from loguru import logger

def s2_extract_and_score(chunk_file, quick_match=False):
    # 需要并行
    logger.info("MathVista: Extract Answers - Start")
    label = 'response'
    logger.info(f"Reading {chunk_file}...")
    results = read_json(chunk_file)
    full_pids = list(results.keys())
    logger.info(f"Number of test problems to run: {len(full_pids)}")
    save_results = []
    for i, pid in enumerate(tqdm(full_pids)):
        # pid = problem['pid']
        item = results[pid]
        assert label in item
        response = item[label]
        extraction = extract_answer(response, item, quick_match)
        results[pid]['extraction'] = extraction
        choices = item['choices']
        question_type = item['question_type']
        answer_type = item['answer_type']
        precision = item['precision']
        answer = item['answer']
        prediction = normalize_extracted_answer(
            extraction,
            choices,
            question_type,
            answer_type,
            precision,
        )
        true_false = safe_equal(prediction, answer)
        item['prediction'] = prediction
        item['true_false'] = true_false
        save_results.append(item)
    save_json(chunk_file, save_results)
    logger.info(f"Saved results to {chunk_file}")
    # logger.info("MathVista: Extract Answers - Finish")

def s4_show_scores(answers_file):
    score_file = answers_file.split(".json")[0] + '_result.json'

    results = read_json(answers_file)
    logger.info("Calculate the average accuracy")
    total = len(results)
    correct = 0
    for pid, item in tqdm(enumerate(results)):
        if item['true_false']:
            correct += 1
    accuracy = correct / total
    scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}
    for pid in tqdm(range(len(results))):
        results[pid].update(results[pid].pop('metadata'))
    results_df = pd.DataFrame(results)
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
    metrics_str = get_full_metrics_str(scores)
    logger.info(metrics_str)
    save_json(score_file, scores)
    logger.info(f"Saved scores to: {score_file}")


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
        result = re.search(r'The answer is "(.*)"\.', response)
        if result:
            response = result.group(1)
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

def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True
