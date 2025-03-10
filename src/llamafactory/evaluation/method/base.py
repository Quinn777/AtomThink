import os
import time
from tqdm import tqdm
from os.path import exists, join
import json
import yaml
import random
from llamafactory.extras.utils import convert_unsupported_types_to_str
from llamafactory.model.loader import load_model
from ..utils.inference_utils import inference_one_step
from ..utils.eval_utils import save_json, read_json, read_jsonl
from ...extras.logging import get_logger
from ...extras.packages import is_pillow_available, is_vllm_available
from ...model import load_tokenizer
from ..utils.datasets import data_map
from ..utils.conversation import conversation_map
from ...data import get_template_and_fix_tokenizer
from datetime import datetime
from PIL import Image
from PIL.Image import Image as ImageObject
if is_pillow_available():
    pass

if is_vllm_available():
    from vllm.lora.request import LoRARequest

def get_dataset(data_args):
    data_path = os.path.join(data_args.dataset_dir, data_args.dataset[0])
    _, ext = os.path.splitext(data_path)
    if ext == ".json":
        all_data = read_json(data_path)
    elif ext == ".jsonl":
        all_data = read_jsonl(data_path)
    else:
        logger.info("Wrong Test Data File!")
        raise
    test_data = all_data
    if data_args.max_samples and isinstance(test_data, list):
        test_data = random.sample(test_data, int(data_args.max_samples))
    elif data_args.max_samples and isinstance(test_data, dict):
        sampled_keys = random.sample(test_data.keys(), int(data_args.max_samples))
        test_data = {key: test_data[key] for key in sampled_keys}
    return test_data

class BaseInference:
    def __init__(self, model_args, data_args, generating_args, finetuning_args):
        self.data_args = data_args
        self.generating_args = generating_args
        self.model_args = model_args
        self.init_path()
        logger.info(f'model_args.model_name_or_path: {model_args.model_name_or_path}')
        tokenizer_module = load_tokenizer(model_args)
        self.tokenizer = tokenizer_module["tokenizer"]
        self.processor = tokenizer_module["processor"]
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.dataset = get_dataset(data_args)
        self.inputs = self.preprocess_data()

        self.model = load_model(self.tokenizer, model_args, finetuning_args)
        self.results = [] if isinstance(self.dataset, list) else {}

    def preprocess_data(self):
        input_dicts = []
        for idx, sample in enumerate(tqdm(self.dataset if isinstance(self.dataset, list) else self.dataset.items())):
            separate_eval, question, images = data_map[self.data_args.dataset_name](self.data_args, sample)
            images = self.template.mm_plugin._regularize_images(images,
                                                                image_resolution=getattr(self.processor,
                                                                                         "image_resolution",
                                                                                         768 * 768))
            input_prompt = conversation_map[self.data_args.template](question,
                                                                     "image",
                                                                     model_name_or_path=self.model_args.model_name_or_path,
                                                                     processor=self.processor,
                                                                     tokenizer=self.tokenizer,
                                                                     images=images)
            if len(images) == 0:
                logger.debug("no image data")
                images = None
            inputs = self.processor(
                text=[input_prompt],
                images=images,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            input_dicts.append({
                "inputs": inputs,
                "prompt": input_prompt,
                "images": images,
                "metadata": sample,
                "separate_eval": separate_eval
            })
        return input_dicts

    def init_path(self):
        if not os.path.isfile(self.data_args.answers_file):
            if not exists(self.data_args.answers_file):
                os.makedirs(self.data_args.answers_file)
            dir_name = self.data_args.answers_file
            exp_name = (f"{self.data_args.dataset_name}_{self.data_args.max_samples}_{self.generating_args.method}_"
                         f"{self.data_args.prompt}_"
                        f"{self.data_args.separate_eval}_"
                        f"{self.generating_args.temperature}_"
                        f"{self.generating_args.top_p}_"
                         f"{self.generating_args.top_k}_"
                        f"{self.generating_args.atomthink_beam_search_num}_"
                         f"{self.generating_args.candidate_num}")
            self.data_args.answers_file = join(dir_name, f"answers_{exp_name}.json")
            self.data_args.extraction_file = join(dir_name, f"extraction_{exp_name}.json")
            self.data_args.score_file = join(dir_name, f"score_{exp_name}.json")

    def run_inference(self):
        for idx, input_item in enumerate(tqdm(self.inputs)):
            if isinstance(self.results, dict):
                metadata_id, metadata = input_item['metadata'][0], input_item['metadata'][1]
            else:
                metadata = input_item['metadata']
            response = ""
            total_inference_times = 1
            start_time = time.time()
            try:
                response = inference_one_step(input_item['inputs'],
                                              self.model,
                                              self.processor,
                                              self.generating_args,
                                              self.generating_args.temperature
                                              )
            except Exception as e:
                logger.error(input_item['prompt'])
                logger.error(e)
            metadata["response"] = response
            metadata["prompt"] = input_item["prompt"]
            if isinstance(self.results, list):
                self.results.append(metadata)
            else:
                self.results[metadata_id] = metadata
            end_time = time.time()
            logger.info(
                f"Time: {end_time - start_time} | Total Inference Times: {total_inference_times}")
        save_json(self.data_args.answers_file, self.results)
        logger.info(f'Save to {self.data_args.answers_file}')

    def save_and_print_configs(self):
        all_args = {"model_args": self.model_args.__dict__, "data_args": self.data_args.__dict__,
                    "generating_args": self.generating_args.__dict__}
        modified_dict = convert_unsupported_types_to_str(all_args)
        if os.path.isfile(all_args["data_args"]["answers_file"]):
            output_dir = os.path.dirname(all_args["data_args"]["answers_file"])
        else:
            output_dir = all_args["data_args"]["answers_file"]
        output_file = os.path.join(output_dir, "config.yaml")
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(modified_dict, f, default_flow_style=False)
        logger.info("Config:\n%s", json.dumps(modified_dict, indent=4))

