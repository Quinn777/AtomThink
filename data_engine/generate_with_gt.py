import time
import copy
import base64
import json
from os.path import join, exists
import re
from PIL import Image
from openai import OpenAI
import argparse
import io
import os
import multiprocessing
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import random
from ..utils.eval_utils import read_json, save_json
from utils.prompts import prompt_w_gt, prompt_make_wrong, prompt_check, str_check
os.environ['OPENAI_API_KEY'] = ""
MAX_TRY_TIMES = 2
sleep_times = [10, 10, 10, 10, 10]


def construct_msg(question, image, template):
    text = f"{template}{question}"
    msg = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}",
                        "detail": "auto",
                    },
                },
            ],
        }
    ]
    return msg


def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format=image.format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def make_api_call(messages, max_tokens=200, is_json=False, model="gpt-4o", temperature=0.2):
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


def get_response(question, image, template):
    messages = construct_msg(question, image, template)
    response = make_api_call(messages, 3000)
    return response


def format_cot(cot):
    match = re.search(r"(Step \d+:.*)", cot.strip(), flags=re.DOTALL)
    if match:
        cot = match.group(1)
    else:
        return None
    pattern = r"Step (\d+):\s*(.*?)\s*(?=Step \d+:|$)"
    steps = re.findall(pattern, cot, re.DOTALL)
    # Format the steps back into a list with the step number and content combined
    formatted_steps = [f"Step {num}: {content.strip()}" for num, content in steps]
    return formatted_steps


def construct_pos_sample(question, cot):
    results = []
    pos_sample = {'question': question, 'process': None, 'label': ['+']}
    for i in range(1, len(cot) + 1):
        partial_string = "\n\n".join(cot[:i]).strip() + "\n\n\n\n\n"
        sample = copy.deepcopy(pos_sample)
        sample['process'] = partial_string
        results.append(sample)
    return results


def construct_neg_sample(question, true_cot, wrong_cot):
    results = []
    neg_sample = {'question': question, 'process': None, 'label': ['-']}
    # Ensure correct and incorrect steps have the same length
    num_steps = min(len(true_cot), len(wrong_cot))
    for i in range(1, num_steps + 1):
        # Use incorrect step at index `i-1` in this version
        mixed_steps = true_cot[:i - 1] + [wrong_cot[i - 1]]
        # Join the steps with newline separators
        mixed_string = "\n\n".join(mixed_steps).strip() + "\n\n\n\n\n"
        sample = copy.deepcopy(neg_sample)
        sample['process'] = mixed_string
        results.append(sample)
    return results


def merge_sft(process_dir, target_path):
    if not exists(process_dir):
        print("No dir!")
        return
    else:
        file_list = os.listdir(process_dir)
        file_list.sort(key=lambda x: int(x.strip('.json')))
        cat_sft_data = []
        for filename in file_list:
            sft_data = read_json(f"{process_dir}/{filename}")
            cat_sft_data.extend(sft_data)
        save_json(target_path, cat_sft_data)
        print(f"{process_dir}:  {len(cat_sft_data)}")
        return len(cat_sft_data), target_path


def merge(process_dir, target_path):
    if not exists(process_dir):
        print("No dir!")
        return
    else:
        files = os.listdir(process_dir)
        total_qa = []
        for file in files:
            f = join(process_dir, file)
            data = read_json(f)
            data['question'] = data['question'].replace("<image>\n", "")
            total_qa.append(data)
        file_name = target_path
        save_json(file_name, total_qa)
        print(f"{process_dir}:  {len(total_qa)}")
        return len(total_qa), file_name


def make_one_data(idx, end_id, samples):
    tic = time.time()
    try_time = 0
    success = False
    while try_time < MAX_TRY_TIMES:
        try:
            data = samples[idx]
            image_file = os.path.join(args.image_dir, data['image'])
            problem_decoded_image = Image.open(image_file)
            base64_image = encode_image_to_base64(problem_decoded_image)
            question = data['conversations'][0]['value']
            gt = data['conversations'][1]['value']
            new_question = "Question: " + question + "\n" + "Reference Answer: " + gt

            # generate true cot
            pos_resp = get_response(new_question, base64_image, prompt_w_gt)
            steps = pos_resp.replace("output: ", "").replace("outputs: ", "").split("\n\n")

            steps = [step.strip("**") for step in steps]
            steps = [step for step in steps if step.startswith("Step ")]
            steps_id = [step.split(":")[0].split("Step ")[1] for step in steps]
            correct_steps_id = [str(index) for index in range(1, len(steps_id) + 1)]
            if steps_id != correct_steps_id:
                print("Format Wrong")
                continue
            pos_resp = "\n\n".join(steps)

            # check cot is true
            check_prompt = str_check.format(question, data['conversations'][1]['value'].split("\nAnswer: ")[1],
                                            pos_resp)
            is_true1 = get_response(check_prompt, base64_image, prompt_check)
            is_true2 = get_response(check_prompt, base64_image, prompt_check)
            if "False" in is_true1 or "false" in is_true1 or "False" in is_true2 or "false" in is_true2:
                print("Not True")
                continue

            # save sft data
            sft_data = []
            final_input_prompt = "<image>\n" + question + "\nAnswer the question using a single word or phrase."
            task_prompt = "\n\nYour task is to predict the next step of reasoning or calculation based on THE GIVEN QUESTION and HISTORICAL REASONING STEPS. Ensure your prediction is a single atomic reasoning step, which should be small and focused. If the historical reasoning steps have already reached a conclusion, there is no need to predict the next step in reasoning; simply reply with \"To sum up, the final answer is: ...\"."
            steps = pos_resp.split("\n\n")

            for step_id in range(len(steps) - 1, -1, -1):
                historical_steps = '\n\n'.join(steps[:step_id])
                requests = f"{final_input_prompt}\n\nHISTORICAL REASONING STEPS:\n{historical_steps}{task_prompt}"
                sft_step_data = {
                    "final_input_prompt": final_input_prompt,
                    "image": data["image"],
                    "conversations": [
                        {
                            "from": "human",
                            "value": requests,
                        },
                        {
                            "from": "gpt",
                            "value": steps[step_id],
                        }
                    ],
                    "answer": data['conversations'][1]['value'].split("\nAnswer: ")[1]
                }
                sft_data.append(sft_step_data)
            save_json(join(args.sft_cot_dir, f"{idx}.json"), sft_data)

            # save true cot
            pos_step_list = format_cot(pos_resp)
            pos_res = copy.deepcopy(data)
            pos_res['true_steps'] = pos_step_list
            save_json(join(args.pos_cot_dir, f"{idx}.json"), pos_res)

            # generate wrong cot
            neg_resp = get_response(pos_resp, base64_image, prompt_make_wrong)

            # save wrong cot
            neg_step_list = format_cot(neg_resp)
            neg_res = copy.deepcopy(pos_res)
            neg_res['wrong_steps'] = neg_step_list

            save_json(join(args.neg_cot_dir, f"{idx}.json"), neg_res)

            # prm format data generating ...
            pos_samples = construct_pos_sample(question, pos_step_list)
            neg_samples = construct_neg_sample(question, pos_step_list, neg_step_list)
            for i, sample in enumerate(pos_samples):
                save_json(join(args.train_data_dir, f"{idx}_pos_{i}.json"), sample)
            for i, sample in enumerate(neg_samples):
                save_json(join(args.train_data_dir, f"{idx}_neg_{i}.json"), sample)
            success = True
            break
        except Exception as e:
            print(f"index {idx}, failed because {e}")
            try_time += 1
            time.sleep(sleep_times[try_time])
            print("retry {}/{}".format(try_time, MAX_TRY_TIMES))
    toc = time.time()
    if success:
        print("[{}]/[{}] Done in {:.2f} seconds".format(idx, end_id, toc - tic))
    else:
        print("[{}]/[{}] Failed. {}".format(idx, end_id, samples[idx]))


def run_parallel(args):
    test_data = read_json(args.input_file)
    random.seed(42)
    test_data = random.sample(test_data, 100)
    if not exists(args.pos_cot_dir):
        os.makedirs(args.pos_cot_dir)
    if not exists(args.neg_cot_dir):
        os.makedirs(args.neg_cot_dir)
    if not exists(args.train_data_dir):
        os.makedirs(args.train_data_dir)
    if not exists(args.sft_cot_dir):
        os.makedirs(args.sft_cot_dir)
    num_workers = multiprocessing.cpu_count()
    process_func = partial(make_one_data,
                           end_id=len(test_data),
                           samples=test_data,
                           )
    with ThreadPoolExecutor(num_workers) as exe:
        exe.map(process_func, list(range(0, len(test_data))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        default="annotations.json")
    parser.add_argument("--image_dir", type=str, default="images")
    parser.add_argument("--pos_cot_dir", type=str,
                        default="pos_cot_dir")
    parser.add_argument("--neg_cot_dir", type=str,
                        default="neg_cot_dir")
    parser.add_argument("--prm_data_dir", type=str,
                        default="prm_data_dir")
    parser.add_argument("--sft_cot_dir", type=str,
                        default="sft_cot_dir")
    args = parser.parse_args()

    run_parallel(args)
    merge(args.prm_data_dir, "prm_data.json")
    merge_sft(args.sft_cot_dir, "sft_data.json")
