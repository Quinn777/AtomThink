import math
import subprocess
import argparse
import os
from loguru import logger
from os.path import join, dirname, abspath, isdir, isfile
from os import listdir, mkdir, makedirs
import sys
import random
current_dir = dirname(os.path.abspath(__file__))
sys.path.append(dirname(dirname(current_dir)))
sys.path.append(dirname(dirname(dirname(current_dir))))
print(sys.path)
print(os.getcwd())
from llamafactory.evaluation.utils.eval_utils import read_json, save_json, save_jsonl, read_jsonl, load_yaml, save_yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_rank', type=int, default=0, help="node rank for inference")
    parser.add_argument('--total_gpus', type=int, default=32, help="total gpu number")
    parser.add_argument('--nproc_per_node', type=int, default=8, help="gpu number per node")
    parser.add_argument('--tasks_per_gpu', type=int, default=1, help="task per gpu")
    parser.add_argument('--config', type=str, default="configs/inference/test.yaml", help="use training config to inference")
    parser.add_argument('--output_dir', type=str, default=None, help="use specific model path to inference")
    parser.add_argument('--remote_work_dir', type=str, default=None, )
    parser.add_argument('--muti_gpu_per_task', type=bool, default=False, help="True when use prm")
    parser.add_argument('--task', type=str, default="MathVista")
    parser.add_argument('--prompt', type=str, default="base")
    parser.add_argument('--method', type=str, default="base", choices=["base", "quick", "slow"])
    parser.add_argument('--separate_eval', type=bool, default=True)
    parser.add_argument('--max_sampling_count', type=int, default=300, help="max sampling count in AtomThink inference")
    parser.add_argument('--max_samples', type=int, default=None, help="sampling the test data")
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--atomthink_beam_search_num', type=int, default=2)
    parser.add_argument('--candidate_num', type=int, default=3)

    args = parser.parse_args()
    return args

def get_task_config(task):
    if task == "MathVista":
        task_config = "configs/inference/mathvista.yaml"
    elif task == "MathVerse":
        task_config = "configs/inference/mathverse.yaml"
    elif task == "MathVision":
        task_config = "configs/inference/mathvision.yaml"
    elif task == "HLE":
        task_config = "configs/inference/hle.yaml"
    else:
        return None
    task_config = load_yaml(task_config)
    return task_config

def parallel_dataset(args, task_config):
    curPath=os.getcwd()
    dir_path = os.path.join(curPath + f"/data/test_data/parallel/{args.task}")
    # print(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    question_file = join(task_config["dataset_dir"], task_config["dataset"])
    if "jsonl" in question_file:
        data = read_jsonl(question_file)
    else:
        data = read_json(question_file)
    if args.max_samples and isinstance(data, list):
        data = random.sample(data, int(args.max_samples))
    elif args.max_samples and isinstance(data, dict):
        sampled_keys = random.sample(data.keys(), int(args.max_samples))
        data = {key: data[key] for key in sampled_keys}
    total_data = len(data)
    logger.info(f"load dataset from {question_file}")
    logger.info(f"dataset len: {total_data}")

    data_per_task = math.ceil(total_data / (args.total_gpus * args.tasks_per_gpu))
    start_index = args.node_rank * args.nproc_per_node * data_per_task * args.tasks_per_gpu
    for gpu_id in range(args.nproc_per_node):
        for task_id in range(args.tasks_per_gpu):
            end_index = min(start_index + data_per_task, total_data)
            if isinstance(data, list):
                task_test_data = data[start_index: end_index]
            elif isinstance(data, dict):
                task_test_data = {k: data[k] for k in list(data)[start_index: end_index]}
            else:
                raise ValueError
            data_name = os.path.join(dir_path, str(args.node_rank) + "_" + str(gpu_id) + "_" + str(task_id) + ".json")
            save_json(data_name,task_test_data)
            start_index = end_index
            if start_index >= total_data:
                break
    logger.info(f"spilt dataset {question_file} done")

def parallel_config(args, task_config, model_config):
    inference_dir = model_config["output_dir"]
    inference_dir = join(join(inference_dir, "inference"), args.task)
    remote_answers_dir = join(model_config['remote_work_dir'], "inference", args.task)
    model_name_or_path = model_config["output_dir"]
    if not os.path.isdir(inference_dir):
        os.makedirs(inference_dir)

    org_config = task_config
    curPath=os.getcwd()
    dir_path = os.path.join(curPath, "configs", "parallel", args.task)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    for gpu_id in range(args.nproc_per_node):
        for task_id in range(args.tasks_per_gpu):
            save_config = org_config
            dataset = os.path.join(args.task, str(args.node_rank) + "_" + str(gpu_id) + "_" + str(task_id) + ".json")
            answers_file = os.path.join(inference_dir, str(args.node_rank) + "_" + str(gpu_id) + "_" + str(task_id) + "/")
            remote_answers_file = os.path.join(remote_answers_dir, str(args.node_rank) + "_" + str(gpu_id) + "_" + str(task_id) + "/")
            save_config["dataset_dir"] = "data/test_data/parallel"
            save_config["dataset"] = dataset
            save_config["answers_file"] = answers_file
            save_config["remote_answers_file"] = remote_answers_file
            save_config['model_name_or_path'] = model_name_or_path
            save_config["max_samples"] = None
            save_config["prompt"] = args.prompt
            save_config["method"] = args.method
            save_config["separate_eval"] = args.separate_eval
            save_config["temperature"] = args.temperature
            save_config["image_resolution"] = model_config["image_resolution"]
            save_config["template"] = model_config["template"]
            save_config["max_sampling_count"] = args.max_sampling_count
            save_config["atomthink_beam_search_num"] = args.atomthink_beam_search_num
            save_config["candidate_num"] = args.candidate_num

            if args.muti_gpu_per_task:
                logger.info("Policy model in cuda:0; reward model in cuda:1")
                save_config["device_map"] = "cuda:0"
                save_config["prm_device_map"] = "cuda:1"
            else:
                logger.info("Policy model in cuda:0; reward model in cuda:0")
                save_config["device_map"] = "cuda:0"
                save_config["prm_device_map"] = "cuda:0"
            save_yaml(save_config, os.path.join(dir_path, str(args.node_rank) + "_" + str(gpu_id) + "_" + str(task_id) + ".yaml"))
    logger.info(f"spilt config in {dir_path} done")

def main():
    args = parse_args()
    print(args)
    if args.config:
        model_config = load_yaml(args.config)
        if args.output_dir != None:
            model_config["output_dir"] = args.output_dir
            logger.warning("Use output_dir in args replace output_dir in config")
    else:
        model_config = {
            "output_dir": args.output_dir,
            "remote_work_dir": args.remote_work_dir,
        }
    log_dir = join(model_config["output_dir"], "logs")
    if not isdir(log_dir):
        makedirs(log_dir, exist_ok=True)
    task_config = get_task_config(args.task)

    if args.muti_gpu_per_task:
        assert args.method == "slow"
        args.total_gpus = int(args.total_gpus / 2)
        args.nproc_per_node = int(args.nproc_per_node / 2)
    parallel_dataset(args, task_config)
    parallel_config(args, task_config, model_config)
    for gpu_id in range(args.nproc_per_node):
        for task_id in range(args.tasks_per_gpu):
            # 生成的命令
            if args.muti_gpu_per_task:
                gpu_env = f"CUDA_VISIBLE_DEVICES={gpu_id * 2},{(gpu_id * 2) + 1}"
            else:
                gpu_env = f"CUDA_VISIBLE_DEVICES={gpu_id}"
            log_file = f"{log_dir}/node{args.node_rank}_gpu_{gpu_id}_{task_id}_task.log"
            config_file = f"configs/parallel/{args.task}/{args.node_rank}_{gpu_id}_{task_id}.yaml"
            command = f"{gpu_env} python src/llamafactory/evaluation/run_eval.py {config_file} 2>&1 | tee -a {log_file}"
            logger.info(f"Starting task: node_{args.node_rank}_gpu_{gpu_id}_task_{task_id}")
            logger.info(command)
            process = subprocess.Popen(command, shell=True, executable="/bin/bash")
    logger.info(f'{args.task} finished!')

if __name__ == '__main__':
    main()
