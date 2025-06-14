import re
import random
import requests


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def find_last_box_start_end_pos(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None, None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None, None
    else:
        return idx, right_brace_idx + 1


def find_maj_ans(string):
    if len(string) == 0:
        return ""

    candidate_answers = []
    while True:
        start_pos, end_pos = find_last_box_start_end_pos(string)
        if start_pos is None:
            break
        candidate_answers.append(
            remove_boxed(string[start_pos:end_pos]).strip()
        )
        string = string[:start_pos]
    candidate_answers = [ans for ans in candidate_answers if ans is not None][::-1]
    if len(candidate_answers) == 0:
        return string
    # costruct (count, order)
    ans_to_count = {}
    ans_to_last_order_index = {}
    ans_info_list = []
    for order_index, ans in enumerate(candidate_answers):
        if ans not in ans_to_count:
            ans_to_count[ans] = 0
        ans_to_count[ans] += 1
        ans_to_last_order_index[ans] = order_index
    for ans in ans_to_count.keys():
        ans_info_list.append({
            "value": ans,
            "count": ans_to_count[ans],
            "last_order_index": ans_to_last_order_index[ans]
        })
    ans_info_list.sort(
        key=lambda x: (x["count"], x["last_order_index"]),
        reverse=True
    )
    return ans_info_list[0]["value"]



def get_answer_str(s: str) -> str:
    res = remove_boxed(last_boxed_only_string(s))
    if res is not None:
        return res
    return s


def solution2answer(solution: str, math_mode="eval_peeking") -> str:
    answer = solution
    if math_mode == "eval_peeking":
        answer = get_answer_str(solution)
    else:
        raise ValueError(f"Invalid math_mode: {math_mode}")
    return answer


def compute_score(solution_str, ground_truth, data_source, extra_info=None):
    # Evaluate equation
    score = None
    try:
        if extra_info is not None and extra_info.get("find_maj", False):
            str1 = find_maj_ans(solution_str)
        else:
            str1 = solution2answer(solution_str)
        str2 = solution2answer(ground_truth)
        # only choice question for gpqa
        if data_source == "gpqa" and len(str1.strip()) > 0:
            str1 = str1.strip()[0]
        if str1 == str2:
            score = 1
        elif len(str1) > 30:
            score = 0
        else:
            # request math eval
            url = "http://127.0.0.1:6284/is_equal"

            # 构造要对比的两个 LaTeX 字符串示例
            data = {
                "str1": str1,
                "str2": str2,
            }
            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()
            result = response.json()
            score = result["equal"]
    except Exception as e:
        print(e)
        score = 0
    if random.randint(1, 500) == 1:
        print(f"--------------------------------")
        print(f"solution_str: {solution_str}")
        print(f"ground_truth: {ground_truth}")
        print(f"score: {score}")
    return score