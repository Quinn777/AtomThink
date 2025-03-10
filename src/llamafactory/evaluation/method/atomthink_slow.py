import time
from tqdm import tqdm
from ...extras.logging import get_logger
from ..utils.inference_utils import inference_one_step, update_inputs_from_rollout, txt_verifier
from ..utils.eval_utils import save_json
from ..utils.conversation import conversation_map
from .base import BaseInference
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import torch.nn.functional as F  # NOTE add for qwen prm
import copy
import heapq

logger = get_logger(__name__)


class AtomThinkSlow(BaseInference):
    def __init__(self, model_args, data_args, generating_args, finetuning_args):
        super().__init__(model_args, data_args, generating_args, finetuning_args)
        self.prm_model, self.prm_tokenizer, self.candidate_tokens, self.step_tag_id = self.load_reward_model(
            self.model_args)

    def run_inference(self):
        for idx, input_item in enumerate(tqdm(self.inputs)):
            if isinstance(self.results, dict):
                metadata_id, metadata = input_item['metadata'][0], input_item['metadata'][1]
            else:
                metadata = input_item['metadata']
            beam_search_num = self.generating_args.atomthink_beam_search_num
            candidate_num = self.generating_args.candidate_num
            rollouts = [[] for i in range(beam_search_num)]
            rollouts_scores = [0 for i in range(beam_search_num)]
            sampling_count = 1
            response = ""
            end = False
            ans_rollout_id = None

            start_time = time.time()
            if input_item['separate_eval']:
                response = inference_one_step(input_item['inputs'],
                                              self.model,
                                              self.processor,
                                              self.generating_args,
                                              self.generating_args.temperature
                                              )
            else:
                try:
                    init_prompt = copy.deepcopy(input_item["prompt"])
                    temp_input_item = copy.deepcopy(input_item)
                    while not end and sampling_count < self.generating_args.max_sampling_count:
                        candidate_list = []
                        for i in range(beam_search_num):
                            try:
                                rollout = rollouts[i]
                            except IndexError:
                                logger.info(rollouts)
                                break
                            if rollout:
                                last_response = copy.copy(rollout[-1])
                                temp_input_item = update_inputs_from_rollout(copy.deepcopy(input_item), rollout,
                                                                             self.processor)
                                if len(temp_input_item['inputs'].data['input_ids'][0]) > self.data_args.cutoff_len:
                                    continue
                            else:
                                last_response = ""
                            # temperature = 0
                            temperature = self.generating_args.temperature
                            for c in range(candidate_num):
                                candidate_c = inference_one_step(temp_input_item['inputs'],
                                                                 self.model,
                                                                 self.processor,
                                                                 self.generating_args,
                                                                 temperature
                                                                 )
                                temperature += 0.5
                                if temperature > 1.0:
                                    temperature = 1.0
                                is_valid, is_final_answer = txt_verifier(candidate_c, last_response)
                                sampling_count += 1
                                if candidate_c:
                                    candidate_roll = rollout + [candidate_c]
                                    if self.model_args.prm_model_type == "qwen_math_prm":
                                        candidate_c_score = self.reward_qwen(init_prompt, candidate_roll)
                                    else:
                                        candidate_c_score = self.reward(init_prompt, candidate_roll)
                                    candidate_list.append([i, candidate_c_score, candidate_roll, is_final_answer])
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
                    response = ans_rollout[-1]  # final answer
                    if "the final answer is: " not in response \
                            and "The final answer is: " not in response \
                            and "To sum up," not in response \
                            and "The answer is" not in response:
                        response = self.fail_try(input_item)
                except Exception as e:
                    logger.info(e)
            end_time = time.time()
            metadata["response"] = response
            metadata["prompt"] = input_item["prompt"]
            metadata["steps"] = rollouts
            metadata["total_inference_times"] = sampling_count
            if isinstance(self.results, list):
                self.results.append(metadata)
            else:
                self.results[metadata_id] = metadata
            logger.info(
                f"Time: {end_time - start_time} | Total Inference Times: {sampling_count}")
        save_json(self.data_args.answers_file, self.results)

    def fail_try(self, input_item):
        question = input_item['prompt'].split("THE GIVEN QUESTION:\n")[-1].split("HISTORICAL REASONING STEPS:")[
            0].replace("\n", " ")
        question += "\nAnswer the question using a single word or phrase."
        input_prompt = conversation_map[self.data_args.template](question,
                                                                 "image",
                                                                 model_name_or_path=self.model_args.model_name_or_path,
                                                                 processor=self.processor,
                                                                 tokenizer=self.tokenizer,
                                                                 images=input_item['images']
                                                                 )
        logger.info(input_prompt)
        input_ids = self.processor(
            text=[input_prompt],
            images=input_item['images'],
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        response = inference_one_step(input_ids,
                                      self.model,
                                      self.processor,
                                      self.generating_args,
                                      self.generating_args.temperature
                                      )
        return response

    @staticmethod
    def load_reward_model(model_args):
        good_token = '+'
        bad_token = '-'
        step_tag = '\n\n\n\n\n'
        prm_model_pretrained = model_args.prm_model
        print(prm_model_pretrained)
        tokenizer = AutoTokenizer.from_pretrained(
            prm_model_pretrained,
            add_eos_token=False, )
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"

        if model_args.prm_model_type == "qwen_math_prm":
            candidate_tokens = [None, None]
            step_tag_id = None
            model = AutoModel.from_pretrained(
                prm_model_pretrained,
                device_map=model_args.prm_device_map,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).eval()
        else:
            candidate_tokens = tokenizer.encode(f" {good_token} {bad_token}")  # [488, 481]
            step_tag_id = tokenizer.encode(f" {step_tag}")[-1]  # 76325
            model = AutoModelForCausalLM.from_pretrained(
                prm_model_pretrained,
                device_map=model_args.prm_device_map,
                torch_dtype=torch.float16)
        return model, tokenizer, candidate_tokens, step_tag_id

    def reward_qwen(self, init_prompt, rollout, caption=None, ):
        question = init_prompt.split("THE GIVEN QUESTION:\n")[-1].split("HISTORICAL REASONING STEPS:")[0].replace("\n", "")
        if caption:
            question = question + " " + caption

        system = "Please reason step by step, and put your final answer within \\boxed{}."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "<extra_0>".join(rollout) + "<extra_0>"},
        ]
        conversation_str = self.prm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        input_ids = self.prm_tokenizer.encode(
            conversation_str,
            return_tensors="pt",
        ).to(self.model_args.prm_device_map)
        with torch.no_grad():
            outputs = self.prm_model(input_ids=input_ids)
        step_sep_id = self.prm_tokenizer.encode("<extra_0>")[0]
        token_masks = (input_ids == step_sep_id)
        probabilities = F.softmax(outputs[0], dim=-1)  #
        probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels
        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]  # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
        res = all_scores_res[0]
        if self.generating_args.aggregation == "avg":
            return sum(res) / len(res)
        elif self.generating_args.aggregation == "min":
            return min(res)
        return res[-1]


