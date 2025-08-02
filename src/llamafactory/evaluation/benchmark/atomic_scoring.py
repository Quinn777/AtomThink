import re
from loguru import logger
from llamafactory.evaluation.utils.prompts import demo_prompt_score
from llamafactory.evaluation.utils.eval_utils import read_json, save_json, save_jsonl, read_jsonl, create_test_prompt, \
    make_api_call, score_func
from tqdm import tqdm

MAX_TRY = 5


def eval_func(answers_file, quick_match):
    logger.info("SuperClevr: Extract Answers - Start")
    results = read_json(answers_file)
    total_score = 0
    for i, sample in enumerate(tqdm(results)):
        results[i]['extractions'] = []
        score = 0
        for candidate in sample['candidates']:
            extraction = extract_by_rule(candidate)
            results[i]['extractions'].append(extraction)
            if quick_match:
                score += quick_match_func(extraction, sample['gt'])
            else:
                score += score_func(extraction, sample['question'], sample['gt'])
        results[i]['score'] = score / len(sample['candidates'])
        total_score += score
    logger.info(f"Accuracy - {total_score / len(results)}")
    save_json(answers_file, results)
    logger.info(f"Saved results to {answers_file}")
    logger.info("SuperClevr: Extract Answers - Finish")


def quick_match_func(response, gt):
    try:
        if int(response) == int(gt):
            return 1
    except Exception as e:
        pass
    return 0


def extract_by_rule(response):
    try:
        pattern = r'<answer>\s*(\d+)\s*</answer>'
        match = re.search(pattern, response)
        if match:
            return int(match.group(1))
    except Exception as e:
        pass
    try:
        pattern = r"the final answer is: (.+?)\."
        match = re.search(pattern, response)
        if match:
            return int(match.group(1))
    except Exception as e:
        pass
    try:
        pattern = r"The answer is: (.+?)\."
        match = re.search(pattern, response)
        if match:
            return int(match.group(1))
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