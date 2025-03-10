from latex2sympy2 import latex2sympy
import re
import json
from tqdm import tqdm
import time
from llamafactory.evaluation.utils.eval_utils import read_json, save_json, save_jsonl, read_jsonl, create_test_prompt, make_api_call, score_func, extract_by_rule
import pprint

def s2_extract_and_score(chunk_file, quick_match=False):

    samples = read_json(chunk_file)
    total_score = 0
    for idx, sample in tqdm(enumerate(samples)):
        gt_option = str(sample['answer'])
        if len(sample['options']) > 0:
            gt_answer_value = sample['options'][ord(gt_option) - ord('A')]
            gt = gt_option + '. ' + gt_answer_value
        else:
            gt_answer_value = ''
            gt = str(sample['answer'])
        model_answer = sample['response'].strip()
        if 'extraction' not in sample.keys() and quick_match:
            for c in 'ABCDEFG':
                if model_answer.endswith(f" {c}.") or model_answer.endswith(
                        f" ({c}).") or model_answer.startswith(
                        f"{c}\n") or model_answer.startswith(f"({c})\n") or model_answer.startswith(
                    f"({c}) {c}\n"):
                    model_answer = c
            if is_number(model_answer.split('is ')[-1].rstrip('.')):
                model_answer = model_answer.split('is ')[-1].rstrip('.')
            if 'oxed{' not in model_answer:
                for flag in ['the final answer is', 'the answer is', 'the correct answer is',
                             'the answer should be']:
                    raw_model_answer = model_answer
                    model_answer = model_answer.split(flag)[-1].strip()
                    if flag in raw_model_answer:
                        model_answer = model_answer.split('\n')[0].split('. ')[0]
                    flag = flag.replace('the', 'The')
                    raw_model_answer = model_answer
                    model_answer = model_answer.split(flag)[-1].strip()
                    if flag in raw_model_answer:
                        model_answer = model_answer.split('\n')[0].split('. ')[0]
            elif model_answer.count('oxed{') > 1:
                model_answer = '\\boxed{' + model_answer.split('oxed{')[-1]

            model_answer = find_math_answer(model_answer).replace('(a)', 'a').replace('(b)', 'b').replace('(c)','c').replace(
                '(d)', 'd').replace('(e)', 'e').replace('{a}', 'a').replace('{b}', 'b').replace('{c}','c').replace(
                '{d}', 'd').replace('{e}', 'e').rstrip('.').lstrip(':').strip()
            samples[idx]['extraction'] = model_answer
            if is_equal(gt_option, model_answer) or is_equal(gt_answer_value, model_answer):
                samples[idx]['score'] = 1
            else:
                samples[idx]['score'] = 0
            total_score = total_score + samples[idx]['score']
        elif 'extraction' not in sample.keys() and not quick_match:
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
                    gt = str(sample['answer'])
                else:
                    formatted_options = []
                    for letter, option in zip(letters, sample['options']):
                        formatted_options.append(f"{letter}. {option};")
                    options = "\n".join(formatted_options)[:-1]
                    question = question + "\nChoices:\n" + options
            question = re.sub(r'<image\d+>\n?', '', question)
            question = '<image>' + question
            extraction = extract_by_rule(model_answer)
            score = score_func(extraction, question, gt)
            total_score += score
            samples[idx]['extraction'] = extraction
            samples[idx]['score'] = score
    print(f"Accuracy - {total_score / len(samples)}")
    print(f"Saving in {chunk_file}")
    save_json(chunk_file, samples)


def s4_show_scores(answer_file):
    samples = read_json(answer_file)

    results_dict = {}
    for idx, sample in tqdm(enumerate(samples), desc='math_level_subject_acc'):
        correct = True if sample['score']==1 else False
        subject = sample['subject']
        level = sample['level']
        for key in [
            '-all',
            f'-level{level}',
            f'{subject}',
            f'{subject}_level{level}',
            f'-level{level}_{subject}'
        ]:
            if key not in results_dict:
                results_dict[key] = [0, 0]
            results_dict[key][0] += 1 if correct else 0
            results_dict[key][1] += 1
    for key in results_dict.keys():
        if results_dict[key][1] == 0:
            results_dict[key] = f'{results_dict[key][0]}/{results_dict[key][1]}=0'
        else:
            results_dict[
                key] = f'{results_dict[key][0]}/{results_dict[key][1]}={round(results_dict[key][0] / max(results_dict[key][1], 1) * 100, 2)}%'

    results_dict = {key: results_dict[key] for key in sorted(results_dict.keys())}
    pprint.pprint(results_dict)
    print(answer_file, ':\t', results_dict['-all'])
    json.dump(results_dict, open(answer_file.replace('.json', '_result.json'), 'w'), indent=4, ensure_ascii=False)

def timestamp() -> str:
    nowtime = time.strftime('-%Y%m%d-%H%M', time.localtime(time.time()))
    print(nowtime)
    return nowtime

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def eval_tuple(s):
    sl = s[1:-1].split(',')

    try:
        # Check if string is a tuple representation and has more than one element
        if s[0] == '(' and s[-1] == ')' and len(sl) > 1:
            s = ','.join([str(round(eval(str(latex2sympy(sub))), 2))
                          if 'infty' not in sub and sub not in ['a', '-a'] else sub for sub in sl])
            return f"({s})"
        elif s[0] == '[' and s[-1] == ']' and len(sl) > 1:
            # Same evaluation process as for tuples
            s = ','.join([str(round(eval(str(latex2sympy(sub))), 2))
                          if 'infty' not in sub and sub not in ['a', '-a'] else sub for sub in sl])
            return f"[{s}]"
    except Exception:  # Catch any exceptions and return the original string
        return s

    # Return original string if it doesn't match tuple or list format
    return s

def is_equal(asw: str, gt_asw: str) -> bool:
    asw = asw.lower()
    gt_asw = gt_asw.lower()

    if asw.replace(' ', '') == '' or gt_asw.replace(' ', '') == '':
        return False
    if gt_asw.strip() == asw.strip():
        return True

    # Convert the string to a tuple format.
    asw = eval_tuple(asw)
    gt_asw = eval_tuple(gt_asw)
    if gt_asw == asw:
        return True

    try:
        if round(eval(str(latex2sympy(gt_asw))), 2) == round(eval(str(latex2sympy(asw))), 2):
            return True

        else:
            return False
    except:
        # If any error occurs during comparison, return False.
        return False


def in_area(id: str, area: str) -> bool:
    if area == 'all':
        return True
    if f'/{area}/' in id or f'{area}_test.csv' in id:
        return True

    # If none of the above conditions are met, return False
    else:
        return False

def extract_nums(s):
    s = s.replace(",", "")
    nums = re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s)
    return_list = []
    for i in range(len(nums)):
        try:
            return_list.append(eval(nums[i].strip().lstrip(" 0")))
        except:
            pass
    return return_list

def find_formula(step):
    assert step.count("<<") == step.count(">>") == 1
    left, right = step.find("<<") + 2, step.find(">>")
    return step[left: right]

def extract_answer(completion):
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        assert False

def delete_extra_zero(n):
    try:
        n = float(n)  # Try to convert the input to a float
    except ValueError:  # If conversion fails
        print("None {}".format(n))  # Print the error message
        return n  # Return the original string
    if isinstance(n, int):
        return str(n)

    # If n is a float after conversion
    if isinstance(n, float):
        n = str(n).rstrip('0')
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)
        return str(n)

def _fix_fracs(string):
    # Split the string based on occurrences of '\frac'.
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        # Exclude the part of the string before the first '\frac'.
        substrs = substrs[1:]

        for substr in substrs:
            new_str += "\\frac"
            # If the current substring already starts with a brace,
            # it's likely formatted correctly.
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string

                a = substr[0]  # Potential numerator.
                b = substr[1]  # Potential denominator.

                # Check if the denominator (b) is already braced.
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    # Check if the string contains exactly one slash, which may indicate it's a fraction.
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")

    try:
        # Try to convert the parts to integers.
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)

        # Convert the fraction to LaTeX representation.
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string

    except:
        return string


def _remove_right_units(string):
    splits = string.split("\\text{ ")

    # Return the part of the string before the last occurrence of "\\text{ ".
    return splits[0]

def _fix_sqrt(string):
    # Check if "\sqrt" is not in the string. If not, return the string as is.
    if "\\sqrt" not in string:
        return string

    # Split the string based on the "\sqrt" substring.
    splits = string.split("\\sqrt")
    new_string = splits[0]

    # Loop through each split portion (after the initial one).
    for split in splits[1:]:
        if len(split) > 0 and split[0] != "{":
            a = split[0]
            # Add braces around the first character and append the rest of the split portion.
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        # Add the new substring to our result string.
        new_string += new_substr

    return new_string

def _strip_string(string):
    # Remove linebreaks
    string = string.replace("\n", "")

    # Remove inverse spaces
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")

    # Replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove degree notation
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # Remove units (assumed to be on the right). Note: The function _remove_right_units is not provided.
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # Handle floating numbers starting with "."
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        string = string.split("=")[-1]
    if len(string.split("\\approx")) == 2:
        string = string.split("\\approx")[-1]
    if 'sqrt' in string:
        string = _fix_sqrt(string)

    # Remove all spaces
    string = string.replace(" ", "")
    if 'sqrt' in string:
        string = _fix_fracs(string)

    # Convert 0.5 to its fraction representation
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)

    return string

def find_math_answer(s: str) -> str:
    s = s.lower()
    if '{}' in s:
        s = s.replace('{}', '')

    try:
        pattern = re.compile('oxed{(.*)}', flags=re.S)
        ans = pattern.findall(s)[-1]
    except:
        ans = s  # If the pattern is not found, consider the entire string as the answer.
    if ans.find('}') != -1 and (ans.find('{') == -1 or ans.find('}') < ans.find('{')):
        ans = ans.split('}')[0]

    # Extract the value after the equals sign or approx symbol.
    ans = ans.split('=')[-1]
    ans = ans.split('\\approx')[-1]
    ans = ans.replace(" ", "").replace("\\,", "").replace('∞', '\\infty')
    ans = ans.replace("+\infty", "\infty").replace("\\\\", "\\").replace("\n", "")
    ans = ans.replace('\\text', '').replace('\\mbox', '').replace('bmatrix', 'pmatrix')
    ans = ans.replace("\\left", "").replace('\\right', '').replace("^{\\circ}", "")
    ans = ans.replace("^\\circ", "").replace("{m}^3", "").replace("m^3", "")
    ans = ans.replace("{units}", "").replace("units", "").replace("{km}", "").replace("km", "")
    return _strip_string(ans)