import os
import fire
import yaml
from pathlib import Path
import sys
from os.path import dirname, isfile
from os.path import join

current_dir = dirname(os.path.abspath(__file__))
sys.path.append(dirname(dirname(current_dir)))
sys.path.append(dirname(dirname(dirname(current_dir))))
print(sys.path)
print(os.getcwd())
from llamafactory.hparams import get_infer_args
from llamafactory.extras.logging import get_logger
from llamafactory.evaluation.method.base import BaseInference
from llamafactory.evaluation.method.atomthink_quick import AtomThinkQuick
from llamafactory.evaluation.method.atomthink_slow import AtomThinkSlow
from llamafactory.evaluation.method.atomic_scoring import AtomicScoring

logger = get_logger(__name__)


def evaluation(config_file):
    args = yaml.safe_load(Path(config_file).read_text())
    model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
    answers_file = data_args.answers_file
    remote_answers_file = data_args.remote_answers_file
    print(answers_file)
    print(remote_answers_file)

    if generating_args.method == "base" or generating_args.method == "cot":
        eval_func = BaseInference(model_args, data_args, generating_args, finetuning_args)
    elif generating_args.method == "quick":
        eval_func = AtomThinkQuick(model_args, data_args, generating_args, finetuning_args)
    elif generating_args.method == "slow":
        eval_func = AtomThinkSlow(model_args, data_args, generating_args, finetuning_args)
    elif generating_args.method == "atomic":
        eval_func = AtomicScoring(model_args, data_args, generating_args, finetuning_args)
    else:
        raise "Unsupported inference method!"

    eval_func.run_inference()


if __name__ == '__main__':
    fire.Fire(evaluation)