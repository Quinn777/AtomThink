import copy
import os
import time
from tqdm import tqdm
from ...extras.logging import get_logger
from ..utils.inference_utils import inference_one_step, add_step_into_hist, txt_verifier
from ..utils.eval_utils import save_json
from ..utils.conversation import conversation_map
from .base import BaseInference

logger = get_logger(__name__)

class AtomicScoring(BaseInference):
    def __init__(self, model_args, data_args, generating_args, finetuning_args):
        super().__init__(model_args, data_args, generating_args, finetuning_args)
    def run_inference(self):
        # print(self.results)
        # print(type(self.results))
        for idx, input_item in enumerate(tqdm(self.inputs)):
            metadata = copy.deepcopy(input_item['metadata'])
            metadata["prompt"] = input_item["prompt"]
            metadata["candidates"] = []
            metadata["steps"] = []
            start_time = time.time()
            for i in range(10):
                resp, steps = self.rollout(input_item)
                metadata['candidates'].append(resp)
                metadata['steps'].append(steps)
                print(resp)
            end_time = time.time()
            self.results.append(metadata)

            logger.info(
                f"Time: {end_time - start_time}")
        print(self.results)
        print(type(self.results))
        save_json(self.data_args.answers_file, self.results)
        logger.info(f'Save to {self.data_args.answers_file}')

    def rollout(self, input_item):
        single_step_sampling_count = 1
        sampling_count = 1
        depth = 1
        steps = []
        last_response = ""
        response = ""
        temperature = self.generating_args.temperature
        try:
            if input_item['separate_eval']:
                response = inference_one_step(input_item['inputs'],
                                              self.model,
                                              self.processor,
                                              self.generating_args,
                                              temperature
                                              )
            else:
                while True:
                    response = inference_one_step(input_item['inputs'],
                                                  self.model,
                                                  self.processor,
                                                  self.generating_args,
                                                  temperature
                                                  )
                    is_valid, is_final_answer = txt_verifier(response, last_response)
                    if is_final_answer:
                        steps.append(response)
                        break
                    elif (sampling_count > self.generating_args.max_sampling_count
                          or len(input_item['inputs'].data['input_ids'][0]) > self.data_args.cutoff_len
                          or depth > self.generating_args.max_depth):
                        print(
                            f"Error: sampling_count： {sampling_count} | tokens：{len(input_item['inputs'].data['input_ids'][0])} | max_depth：{depth}")
                        break
                    elif not is_valid:
                        if single_step_sampling_count > self.generating_args.max_single_step_sampling_count:
                            break
                        else:
                            single_step_sampling_count += 1
                            sampling_count += 1
                            if temperature <= 1.0:
                                temperature += 0.1
                            continue
                    else:
                        input_item = add_step_into_hist(self.processor, input_item, f"\n{response}")
                        steps.append(response)
                        last_response = response
                        sampling_count += 1
                        depth += 1
                        single_step_sampling_count = 1
                if "the final answer is: " not in response and "The final answer is: " not in response and "To sum up," not in response and "The answer is:" not in response:
                    response = self.fail_try(input_item)

        except Exception as e:
            logger.error(e)
        return response, steps

    def fail_try(self, input_item):
        question = input_item['prompt'].split("THE GIVEN QUESTION:\n")[-1].split("HISTORICAL REASONING STEPS:")[0].replace("\n", " ")
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