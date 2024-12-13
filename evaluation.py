# -*- coding: utf-8 -*-
# @Time    : 2024/12/10 15:18
# @Author  : Kun Xiang
# @File    : evaluation.py
# @Software: PyCharm

import argparse
import os
import time
import copy
from tqdm import tqdm
import torch
from os.path import join, exists
from PIL import Image
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import process_images, load_image_from_base64, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from utils.prompts import prompt_template
from utils.inference_utils import inference_one_step, construct_prompt_from_rollout, add_step_into_hist, verifier
from utils.eval_utils import read_mmcv_config, save_json, read_json, read_jsonl
from utils.mathvista_utils import calculate_score_func, extract_answer_func
from utils.mathverse_utils import extract_answer_s1, score_answer_s2, show_scores

from loguru import logger
from datetime import datetime
import random
import re
import heapq
from transformers import AutoTokenizer, AutoModelForCausalLM
from argparse import Namespace

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--model-path", type=str, default=None)
parser.add_argument("--model-base", type=str, default=None)
parser.add_argument('--conv_template', type=str, default="llama3")
parser.add_argument("--mmcv_config", type=str, default=None)

parser.add_argument('--prm', type=str, default=None)
parser.add_argument('--inference_method', type=str, default=None)
parser.add_argument('--score_method', type=str, default=None)
parser.add_argument("--cot_beam_search_num", type=int, default=2)
parser.add_argument("--candidate_num", type=int, default=3)
parser.add_argument("--prm_model", type=str, default=None)
parser.add_argument("--temperature", type=float, default=None)
parser.add_argument("--top_p", type=float, default=None)
parser.add_argument("--top_k", type=int, default=None)
parser.add_argument('--sampling_n', type=int, default=None)
parser.add_argument('--try_n_one_step', type=int, default=None)
parser.add_argument('--try_n_total_step', type=int, default=None)
parser.add_argument('--max_prompt_length', type=int, default=None)
parser.add_argument('--max_step_depth', type=int, default=None)
parser.add_argument("--num_beams", type=int, default=None)

# input data
parser.add_argument("--benchmark", type=str, default=None)
parser.add_argument("--image_dir", type=str, default=None)
parser.add_argument("--question_file", type=str, default=None)
parser.add_argument("--query_file", type=str, default=None)
parser.add_argument('--mmmu_config', type=str, default=None)

parser.add_argument('--split', type=str, default=None)
parser.add_argument('--subset', type=int, default=None)
parser.add_argument('--cache', type=bool, default=False)
parser.add_argument('--trunk_response', type=int, default=-1, help='trunk response to the last n words')
parser.add_argument('--max_num_problems', type=int, default=10000, help='The max number of problems to run')
parser.add_argument('--quick_match', type=bool, default=False)
parser.add_argument('--caculate_gain', action='store_true', help='caculate the socre gains over random guess')
parser.add_argument('--ignore_empty_extractions', action='store_true',
                    help='If true, ignore empty extractions, otherwise process')
parser.add_argument('--random_file', type=str, default='score_random_guess.json')

# output data
parser.add_argument("--answers_file", type=str, default=None)
parser.add_argument('--extraction_file', type=str, default=None)
parser.add_argument('--score_file', type=str, default=None)
parser.add_argument('--log', type=str, default=None)
parser.add_argument("--chunk-idx", type=int, default=0)
parser.add_argument('--seed', type=int, default=42)
random.seed(42)


class Evaluation:
    def __init__(self):
        args = parser.parse_args()
        if args.config:
            config = read_json(args.config)
            self.args = merge_config_and_args(config, args)
        else:
            self.args = args
        self.test_data = self.get_test_data(self.args)
        if not os.path.isfile(self.args.answers_file):
            self.tokenizer, self.model, self.image_processor, self.model_name = self.load_model(self.args)

            if "beam_search" in self.args.inference_method or "best_of_n" in self.args.inference_method:
                self.prm_model, self.prm_tokenizer, self.candidate_tokens, self.step_tag_id = self.load_reward_model(
                    self.args)
        else:
            self.tokenizer, self.model, self.image_processor, self.model_name = None, None, None, "None"

        # mathvista
        if self.args.query_file:
            self.query = read_json(self.args.query_file)
            self.model_responses = {}
        else:
            self.query = None
            self.model_responses = []
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.exp_name = f"{current_time}-{self.args.benchmark}_{self.args.split}_{self.args.prm}_{self.args.inference_method}"
        log_filename = join(self.args.log, f"{self.exp_name}.log")
        logger.add(log_filename, level="INFO")
        logger.info(f"Exp: {self.exp_name}")

    def init_path(self):
        if not os.path.isfile(self.args.answers_file):
            if not os.path.exists(join(self.args.answers_file, self.model_name)):
                os.makedirs(join(self.args.answers_file, self.model_name))
            self.args.answers_file = join(join(self.args.answers_file, self.model_name),
                                          f"answers_{self.exp_name}.json")
            if self.args.benchmark == "MathVision":
                self.args.answers_file = self.args.answers_file.replace(".json", ".jsonl")

        if not os.path.isfile(self.args.extraction_file):
            self.args.extraction_file = join(join(self.args.extraction_file, self.model_name),
                                             f"extraction_{self.exp_name}.json")
        if not os.path.isfile(self.args.score_file):
            self.args.score_file = join(join(self.args.score_file, self.model_name), f"score_{self.exp_name}.json")

    @staticmethod
    def post_prompt_construction_unified(args, question, question_type=None, answer_type=None, precision=None):
        question = re.sub(r"<image\d+>", "", question)
        question = question.replace("<image> ", "").replace(" <image>", "").replace("<image>", "").strip()
        question = prompt_template.format(question, "")
        conv = copy.deepcopy(conv_templates[args.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt

    def run_inference(self):
        if os.path.isdir(self.args.answers_file):
            logger.info(f"{self.args.answers_file} is a dictionary!")
            raise
        for idx, sample in enumerate(tqdm(self.test_data)):
            if "MathVista" in self.args.benchmark:
                sample_id = copy.deepcopy(sample)
                sample: dict = self.test_data[sample_id].copy()
                query = self.query[sample_id].split("\nQuestion: ")[-1]
                prompt = self.post_prompt_construction_unified(self.args, query, sample["question_type"],
                                                               sample["answer_type"], sample["precision"])
            elif "MathVerse" in self.args.benchmark:
                prompt = self.post_prompt_construction_unified(self.args, sample["question_for_eval"],
                                                               sample["question_type"])

            start_time = time.time()
            if 'image' in sample.keys():
                ori_img, image_tensor, image_size = self.preprocess_image(self.args, sample, self.image_processor,
                                                                          self.model.config)
            else:
                ori_img, image_tensor, image_size = None, None, None
            response = ""
            total_inference_times = 1
            steps = []
            step_length = 0

            try:
                if self.args.inference_method == "quick_thinking":
                    response, steps, inference_times = self.quick_thinking(prompt, response, self.model, image_tensor,
                                                                      image_size, self.tokenizer, self.args,)
                    if "the final answer is: " not in response and "The final answer is: " not in response and "To sum up," not in response:
                        response = self.fail_try(prompt, image_tensor, image_size)
                    total_inference_times += inference_times
                    step_length = len(steps)
                elif self.args.inference_method == "slow_thinking":
                    response, steps, inference_times = self.slow_thinking(prompt, image_tensor, image_size)
                    if "the final answer is: " not in response and "The final answer is: " not in response and "To sum up," not in response:
                        response = self.fail_try(prompt, image_tensor, image_size)
                    total_inference_times += inference_times
                    step_length = len(steps[0])
                else:
                    response = inference_one_step(prompt, self.model, image_tensor, image_size, self.tokenizer,
                                                  self.args)
                    step_length = len(steps)
            except Exception as e:
                logger.info(e)
            sample["response"] = response
            sample["prompt"] = prompt
            sample["steps"] = steps
            sample["total_inference_times"] = total_inference_times

            if isinstance(self.model_responses, list):
                self.model_responses.append(sample)
            else:
                self.model_responses[sample_id] = sample
            end_time = time.time()
            logger.info(
                f"Time: {end_time - start_time} | Total Inference Times: {total_inference_times} | Total steps: {step_length}")
            save_json(self.args.answers_file, self.model_responses)

    def evaluate_answer(self):
        if "MathVista" in self.args.benchmark:
            extract_answer_func(self.args.answers_file, self.args.query_file, self.args.max_num_problems,
                                self.args.quick_match)
            calculate_score_func(self.args.answers_file, self.args.max_num_problems, self.args.ignore_empty_extractions,
                                 self.args.caculate_gain, self.args.random_file, self.args.score_file)
        elif "MathVerse" in self.args.benchmark:
            extract_answer_s1(self.args.answers_file, self.args.extraction_file, self.args.trunk_response,
                              self.args.api_key, self.args.cache)
            score_answer_s2(self.args.extraction_file, self.args.score_file, self.args.cache, self.args.api_key,
                            self.args.quick_match)
            show_scores(self.args.score_file)
        else:
            logger.info("Unknown Evaluate Date!")
            raise
        return

    @staticmethod
    def load_model(args):
        mmcv_config = None
        if args.mmcv_config is not None:
            mmcv_config = read_mmcv_config(args.mmcv_config)
            args.conv_mode = mmcv_config.model_args.version
            if mmcv_config.training_args.get('lora_enable', False):
                args.model_base = mmcv_config.model_args.language_model.pretrained_model_name_or_path
            args.model_path = mmcv_config.training_args.output_dir
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        model_base = args.model_base
        if args.mmcv_config is not None:
            tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name,
                                                                                   mmcv_config=mmcv_config,
                                                                                   device="cuda:0")
        else:
            tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name,
                                                                                   device="cuda:0")
        return tokenizer, model, image_processor, model_name

    @staticmethod
    def get_test_data(args):
        _, ext = os.path.splitext(args.question_file)
        if ext == ".json":
            all_data = read_json(args.question_file)
        elif ext == ".jsonl":
            all_data = read_jsonl(args.question_file)
        else:
            logger.info("Wrong Test Data File!")
            raise
        test_data = all_data
        if args.subset and isinstance(test_data, list):
            test_data = random.sample(test_data, int(args.subset))
        elif args.subset and isinstance(test_data, dict):
            sampled_keys = random.sample(test_data.keys(), int(args.subset))
            test_data = {key: test_data[key] for key in sampled_keys}
        return test_data

    @staticmethod
    def preprocess_image(args, sample, image_processor, config):
        max_w, max_h = 0, 0
        images = []
        if isinstance(sample['image'], list):
            for image_file in sample['image']:
                image_file = os.path.join(args.image_dir, image_file)
                image = Image.open(image_file)
                images.append(image.convert('RGB'))
                max_w = max(image.size[0], max_w)
                max_h = max(image.size[1], max_h)
        else:
            image_file = sample['image']
            image_file = os.path.join(args.image_dir, image_file)
            image = Image.open(image_file)
            images.append(image.convert('RGB'))
            max_w = image.size[0]
            max_h = image.size[1]
        image_size = (max_w, max_h)
        if "llava-llama3-8b" in args.model_path:
            image_tensor = [process_images(images, image_processor, config)[0].half().to("cuda:0")]
        else:
            image_tensor = process_images(images, image_processor, config)[0].half().to("cuda:0")
        return image, image_tensor, image_size

    @staticmethod
    def quick_thinking(prompt, response, model, img_list, image_size, tokenizer, args, ):
        try_n_one_step = 1
        try_n_total_step = 1
        total_steps_n = 1
        steps = []
        last_response = copy.copy(response)
        while True:
            response = inference_one_step(prompt, model, img_list, image_size, tokenizer, args)
            is_valid, is_final_answer = verifier(response, last_response)
            if is_final_answer:
                steps.append(response)
                break
            elif (try_n_total_step > args.try_n_total_step
                  or len(prompt) > args.max_prompt_length
                  or total_steps_n > args.max_step_depth):
                print(
                    f"Error: try_n_total_step： {try_n_total_step} | len(prompt)：{len(prompt)} | total_steps_n：{total_steps_n}")
                break
            elif not is_valid:
                if try_n_one_step > args.try_n_one_step:
                    break
                else:
                    try_n_one_step += 1
                    try_n_total_step += 1
                    continue
            else:
                prompt = add_step_into_hist(prompt, f"\n{response}")
                steps.append(response)
                last_response = response
                try_n_total_step += 1
                total_steps_n += 1
                try_n_one_step = 1
        return response, steps, try_n_total_step

    def slow_thinking(self, prompt, img_tensor, image_size):
        init_prompt = copy.deepcopy(prompt)
        beam_search_num = self.args.cot_beam_search_num
        candidate_num = self.args.candidate_num
        rollouts = [[] for i in range(beam_search_num)]
        rollouts_scores = [0 for i in range(beam_search_num)]

        try_n_total_step = 0
        end = False
        ans_rollout_id = None
        while not end and try_n_total_step < self.args.try_n_total_step:
            candidate_list = []
            for i in range(beam_search_num):
                try:
                    rollout = rollouts[i]
                except IndexError:
                    print(rollouts)
                    break
                if rollout:
                    last_response = copy.copy(rollout[-1])
                    prompt = construct_prompt_from_rollout(init_prompt, rollout)
                    if len(prompt) > self.args.max_prompt_length:
                        continue
                else:
                    last_response = ""
                for c in range(candidate_num):
                    candidate_c = inference_one_step(prompt, self.model, img_tensor, image_size, self.tokenizer,
                                                     self.args).strip()
                    is_valid, is_final_answer = verifier(candidate_c, last_response)
                    try_n_total_step += 1
                    if candidate_c:
                        candidate_roll = rollout + [candidate_c]
                        candidate_c_score = self.reward(init_prompt, candidate_roll)
                        candidate_list.append([i, candidate_c_score, candidate_roll, is_final_answer])

            # top n score
            if len(candidate_list) > beam_search_num:
                top_n_candidates = heapq.nlargest(beam_search_num, candidate_list, key=lambda x: x[1])
                for i, candidate in enumerate(top_n_candidates):
                    rollouts[i] = candidate[2]
                    rollouts_scores[i] = candidate[1]
                    if candidate[3] and not end:
                        end = True
                        ans_rollout_id = i
            elif len(candidate_list) > 0:
                top_n_candidates = copy.deepcopy(candidate_list)
                rollouts = []
                rollouts_scores = []
                for i, candidate in enumerate(top_n_candidates):
                    rollouts.append(candidate[2])
                    rollouts_scores.append(candidate[1])
                    if candidate[3] and not end:
                        end = True
                        ans_rollout_id = i
            else:
                end = True
        if ans_rollout_id is None:
            ans_rollout_id = rollouts_scores.index(max(rollouts_scores))
            logger.info("Cannot find final answer!")
        ans_rollout = rollouts[ans_rollout_id]
        ans = ans_rollout[-1]
        return ans, rollouts, try_n_total_step

    def fail_try(self, prompt, image_tensor, image_size):
        question = prompt.split("THE GIVEN QUESTION:\n")[-1].split("HISTORICAL REASONING STEPS:")[0].replace("\n", " ")
        question += "\nAnswer the question using a single word or phrase."
        prompt = "<image>\n" + question
        conv = copy.deepcopy(conv_templates[self.args.conv_template])
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        logger.info(prompt)
        response = inference_one_step(prompt, self.model, image_tensor, image_size, self.tokenizer, self.args)
        return response

    @staticmethod
    def load_reward_model(args):
        good_token = '+'
        bad_token = '-'
        step_tag = '\n\n\n\n\n'
        prm_model_pretrained = args.prm_model
        tokenizer = AutoTokenizer.from_pretrained(
            prm_model_pretrained,
            add_eos_token=False, )
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"
        candidate_tokens = tokenizer.encode(f" {good_token} {bad_token}")  # [488, 481]
        step_tag_id = tokenizer.encode(f" {step_tag}")[-1]  # 76325
        model = AutoModelForCausalLM.from_pretrained(
            prm_model_pretrained,
            device_map="cuda:1",
            torch_dtype=torch.float16)
        return model, tokenizer, candidate_tokens, step_tag_id

    def reward(self, init_prompt, rollout, caption=None, score_method=""):
        question = init_prompt.split("THE GIVEN QUESTION:\n")[-1].split("HISTORICAL REASONING STEPS:")[0].replace("\n",
                                                                                                                  " ")
        if caption:
            input_for_prm = question + " " + caption + " \n\n\n\n\n".join(rollout) + " \n\n\n\n\n"
        else:
            input_for_prm = question + " " + " \n\n\n\n\n".join(rollout) + " \n\n\n\n\n"
        input_id = torch.tensor([self.prm_tokenizer.encode(input_for_prm)]).to('cuda:1')
        with torch.no_grad():
            logits = self.prm_model(input_id).logits[:, :, self.candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0]
            step_scores = scores[input_id == self.step_tag_id]
        res = []
        for step_score in step_scores:
            res.append(round(step_score.item(), 4))
        if "avg" in score_method:
            return sum(res) / len(res)
        elif "min" in score_method:
            return min(res)
        return res[-1]


def merge_config_and_args(config, args):
    merged = config.copy()
    for key, value in vars(args).items():
        default_value = parser.get_default(key)
        if value != default_value:
            merged[key] = value
        elif key not in merged:
            merged[key] = value
    return Namespace(**merged)


if __name__ == '__main__':
    eva = Evaluation()
    eva.init_path()
    logger.info(eva.args)
    if not os.path.isfile(eva.args.answers_file):
        eva.run_inference()
        eva.evaluate_answer()
    else:
        eva.evaluate_answer()