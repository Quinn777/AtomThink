import time
import base64
import json
from os.path import join, exists
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
from generate_with_gt import encode_image_to_base64

os.environ['OPENAI_API_KEY'] = ""
MAX_TRY_TIMES = 2
sleep_times = [10, 10, 10, 10, 10]


def generate_cot_response(question, image):
    messages = [
            {"role": "system", "content": """You are an expert AI assistant that explains your reasoning step by step. Your task is to continue your previous conversation and predict the next step in reasoning. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'content', and 'next_action' (either 'continue' or 'final_answer') keys.

1. Ensure your output is a single atomic reasoning step, which should be small and focused.
2. Ensure that your reasoning incorporates all relevant details from the provided image.
3. Break down your explanation into clear, concise steps. Use as many reasoning steps as possible while avoiding unnecessary or redundant information.
4. In your reasoning process, utilize various approaches to explore the answer comprehensively, ensuring a thorough analysis.
5. Base your reasoning strictly on the information available in the image and prior context to prevent inaccuracies.

Example1 of valid response:
    ```json
    {
        "content": "Step 1: The image shows an isosceles triangle, we need to calculate...",
        "next_action": "continue"
    }```

Example2 of valid response:
    ```json
    {
        "content": "Step 2: Applying the Pythagorean theorem: ...",
        "next_action": "continue"
    }```

Example3 of valid response:
    ```json
    {
        "content": "Step N: We finally calculated...",
        "next_action": "final_answer"
    }```
    """},
        {"role": "user",
             "content": [
                {
                    "type": "text",
                    "text": question
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    }
                }
            ]},
        {"role": "assistant",
             "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
        ]

    steps = []
    step_count = 1

    while True:
        step_data = make_api_call(messages, 300)
        steps.append(step_data['content'])
        messages.append({"role": "assistant", "content": json.dumps(step_data)})
        if step_data['next_action'] == 'final_answer' or step_count > 25:  # Maximum of 25 steps to prevent infinite thinking time. Can be adjusted.
            break
        step_count += 1
    # Generate final answer
    messages.append({"role": "user",
                     "content": "Please provide the final answer based solely on your reasoning above. Do not use JSON formatting. simply reply with \"Step N: To sum up, the final answer is: ...\". N is the step number."})
    final_data = make_api_call(messages, 200, is_final_answer=True)
    steps.append(final_data)
    return steps


def make_api_call(messages, max_tokens, is_final_answer=False, custom_client=None):
    if custom_client != None:
        client = custom_client
    else:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", None)
        )
        client.base_url = 'http://rerverseapi.workergpt.cn/v1'
    for attempt in range(3):
        try:
            if is_final_answer:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.2,
                )
                return response.choices[0].message.content
            else:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return {"content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}",
                            "next_action": "final_answer"}
                else:
                    return {"content": f"Failed to generate step after 3 attempts. Error: {str(e)}",
                            "next_action": "final_answer"}
            time.sleep(1)


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

            # generate true cot
            steps = generate_cot_response(question, base64_image)
            steps = [step.strip("**") for step in steps]
            steps = [step for step in steps if step.startswith("Step ")]
            steps_id = [step.split(":")[0].split("Step ")[1] for step in steps]
            correct_steps_id = [str(index) for index in range(1, len(steps_id) + 1)]
            if steps_id != correct_steps_id:
                print("Format Wrong")
                continue
            resp = "\n\n".join(steps)

            # save sft data
            sft_data = []
            final_input_prompt = "<image>\n" + question + "\nAnswer the question using a single word or phrase."
            task_prompt = "\n\nYour task is to predict the next step of reasoning or calculation based on THE GIVEN QUESTION and HISTORICAL REASONING STEPS. Ensure your prediction is a single atomic reasoning step, which should be small and focused. If the historical reasoning steps have already reached a conclusion, there is no need to predict the next step in reasoning; simply reply with \"To sum up, the final answer is: ...\"."
            steps = resp.split("\n\n")

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
                save_json(join(args.sft_cot_dir, f"{idx}_{step_id}.json"), sft_step_data)

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
    parser.add_argument("--sft_cot_dir", type=str,
                        default="sft_cot_dir")
    args = parser.parse_args()
    run_parallel(args)