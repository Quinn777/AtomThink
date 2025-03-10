from .prompts import atomthink_prompt_template, multimath_prompt, r1v_template, cot_prompt, amath_v3_train_template
from PIL import Image
from PIL.Image import Image as ImageObject
from os.path import join
from .eval_utils import save_json, read_json, read_jsonl
import string
import re

def load_images(data_args, sample):
    images = None
    if "images" in sample.keys():
        images = []
        if sample["images"]:
            for image in sample["images"]:
                if not isinstance(image, (str, ImageObject)):
                    raise ValueError(f"Expected image input is a path or PIL.Image, but got {type(image)}.")
                if isinstance(image, str):
                    image = Image.open(join(data_args.image_dir, image)).convert("RGB")
                images.append(image)
    elif "image" in sample.keys():
        images = []
        if sample["image"]:
            if not isinstance(sample["image"], (str, ImageObject)):
                raise ValueError(
                    f"Expected image input is a path or PIL.Image, but got {type(sample['image'])}.")

            if isinstance(sample["image"], str):
                image = Image.open(join(data_args.image_dir, sample["image"])).convert("RGB")
                images.append(image)
    return images

def mathverse(data_args, sample):
    if not sample['question']:
        question = sample['query_wo']
    elif "Choices" in sample['question']:
        question = sample['question'].replace('Choices', 'Options')
        question_part, choices_part = question.split('Options:\n')
        formatted_choices = ""
        if choices_part:
            choices = choices_part.split('\n')
            index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
            for i, choice in enumerate(choices):
                if not choice:
                    continue
                if ':' in choice:
                    value = ':'.join(choice.split(':')[1:])
                elif '.' in choice:
                    value = '.'.join(choice.split(':')[1:])
                else:
                    print(choice)
                    value = choice
                option = index[i]
                formatted_choices += f"({option.strip()}) {value.strip()}\n"
            formatted_choices = formatted_choices.strip()
        question = question_part + "Options:\n" + formatted_choices
    else:
        question = sample['question_for_eval']
    if data_args.prompt == "base":
        question = question
    elif data_args.prompt == "cot":
        question = sample["query_cot"]
    elif data_args.prompt == "quick" or data_args.prompt == "slow":
        question = atomthink_prompt_template.format(question, "")
    else:
        raise "Unsupported prompt type!"
    images = load_images(data_args, sample)
    return False, question, images

def mathvista(data_args, sample):
    idx, sample = sample
    question = sample['question']
    separate_eval_flag = False
    if sample["precision"]:
        hint = sample['query'].split("\nQuestion: ")[0].replace("Hint: ", "")
        question += f" {hint}"
    if sample['choices']:
        options = list(string.ascii_uppercase)
        result = ""
        for i, value in enumerate(sample['choices']):
            option = options[i]
            result += f"({option}) {value}\n"
        result = result.strip()
        question = sample['question'] + "\nOptions:\n" + result
    if data_args.prompt == "base" or (data_args.separate_eval and sample['metadata']['category'] == 'general-vqa'):
        assert question
        separate_eval_flag = True
    elif data_args.prompt == "cot":
        question = question + "\n" + cot_prompt
    elif data_args.prompt == "quick" or data_args.prompt == "slow":
        question = atomthink_prompt_template.format(question, "")
    else:
        raise "Unsupported prompt type!"
    images = load_images(data_args, sample)
    return separate_eval_flag, question, images


def mathvision(data_args, sample):
    question = sample['question']
    if sample['options']:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if len(sample['options']) > len(letters):
            raise ValueError(f"Too many options for the available letters (A-Z): {sample}")
        flag = True
        for o in sample['options']:
            if o not in letters:
                flag = False
        if flag:
            question = "Answer the following multiple-choice questions by providing only the correct option letter. " + question
        else:
            formatted_options = []
            for letter, option in zip(letters, sample['options']):
                formatted_options.append(f"{letter}. {option};")
            options = "\n".join(formatted_options)[:-1]
            question = question + "\nChoices:\n" + options
    question = re.sub(r'<image\d+>\n?', '', question).replace('<image>', '')
    # question = '<image>' + question
    if data_args.prompt == "base":
        question = question
    elif data_args.prompt == "cot":
        question = question + "\n" + cot_prompt
    else:
        raise "Unsupported prompt type!"

    images = load_images(data_args, sample)
    return False, question, images

def hle(data_args, sample):
    question = sample["question"]
    # print(question)
    if data_args.prompt == "base":
        question = question
    elif data_args.prompt == "cot":
        question = question + "\n" + cot_prompt
    elif data_args.prompt == "quick" or data_args.prompt == "slow":
        question = atomthink_prompt_template.format(question, "")
    else:
        raise "Unsupported prompt type!"
    images = load_images(data_args, sample)
    return False, question, images

data_map = {
    "MathVerse": mathverse,
    "MathVista": mathvista,
    "MathVision": mathvision,
    "HLE": hle,
}