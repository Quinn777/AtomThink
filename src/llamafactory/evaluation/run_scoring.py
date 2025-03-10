import os
import sys
from os.path import dirname, exists, join
import concurrent.futures
import argparse
current_dir = dirname(os.path.abspath(__file__))
sys.path.append(dirname(dirname(current_dir)))
sys.path.append(dirname(dirname(dirname(current_dir))))
print(sys.path)
print(os.getcwd())
from llamafactory.extras.logging import get_logger
from llamafactory.evaluation.utils.eval_utils import s0_merge_answers, s1_separate_n, s3_merge_save_answers, read_json, save_json, save_jsonl, read_jsonl, load_yaml, save_yaml
logger = get_logger(__name__)

def scoring():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default='MathVista', type=str)
    parser.add_argument('--config', type=str, default=None, help="configs/train_full/test.yaml")
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument("--api_process", default=32, type=int)
    parser.add_argument("--tag", default="base_base", type=str)

    args = parser.parse_args()
    if "MathVista" in args.dataset_name:
        from llamafactory.evaluation.benchmark.mathvista import s2_extract_and_score, s4_show_scores
    elif "MathVerse" in args.dataset_name:
        from llamafactory.evaluation.benchmark.ori_mathverse import s2_extract_and_score, s4_show_scores
    elif "MathVision" in args.dataset_name:
        from llamafactory.evaluation.benchmark.mathvision import s2_extract_and_score, s4_show_scores
    elif "HLE" == args.dataset_name:
        from llamafactory.evaluation.benchmark.hle import s2_extract_and_score, s4_show_scores
    else:
        logger.info("Unknown Evaluate Data!")
        raise
    if args.config:
        model_output_dir = load_yaml(args.config)['output_dir']
    else:
        model_output_dir = args.output_dir
    print(model_output_dir)

    answers_file = s0_merge_answers(model_output_dir, args.dataset_name, args.tag)
    answers_file_list = s1_separate_n(answers_file, args.api_process)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(s2_extract_and_score, answers_file_list)
    s3_merge_save_answers(answers_file, answers_file_list)
    s4_show_scores(answers_file)

if __name__ == '__main__':
    scoring()