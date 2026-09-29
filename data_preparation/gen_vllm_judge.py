"""vLLM judge generation and verdict parsing utilities."""
import re
from typing import Optional

import numpy as np
import torch
from jinja2 import TemplateError
from vllm import SamplingParams


def check_correct(flip,gens):
    corrects = []
    for gen in gens:
        if "[[A]]" in gen or "[A]" in gen or '\\boxed{Assistant 1}' in gen:
            if flip==1:
                corrects.append(0)
            else:
                corrects.append(1)
        elif "[[B]]" in gen or "[B]" in gen or '\\boxed{Assistant 2}' in gen:
            if flip==1:
                corrects.append(1)
            else:
                corrects.append(0)
        else:
            corrects.append(0)
    return corrects



@torch.no_grad()
def generate_response_vllm(dataset,model,model_name,tokenizer,sys_prompt,user_template,local_index,n=1,chosen_name = 'chosen', rejected_name = 'rejected', seed=42):
    with torch.inference_mode():

        temp = 0
        max_tokens = 4096
        sampling_params = SamplingParams(temperature=temp, top_p=1.0, n=n, max_tokens=max_tokens,stop_token_ids=[tokenizer.eos_token_id])

        all_prompt = dataset['prompt'].tolist()
        all_chosen = dataset[chosen_name].tolist() # A response
        all_rejected = dataset[rejected_name].tolist() # B response
        chat_prompts = []
        flips = []
        flip_rng = np.random.default_rng(seed + 100000 * local_index)

        for i in range(len(all_chosen)):
            chosen = all_chosen[i]
            rejected = all_rejected[i]
            prompt = all_prompt[i]

            sample = flip_rng.integers(0, 2)
            if sample==1:
                answer_a = chosen
                answer_b = rejected
                flips.append(0)
            elif sample==0:
                answer_a = rejected
                answer_b = chosen
                flips.append(1)
            else:
                raise ValueError("sample should be 0 or 1")


            user_content = user_template.format(question=prompt, answer_a=answer_a, answer_b=answer_b)

            try:
                prompt_message = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
                ]
                chat_text = tokenizer.apply_chat_template(
                    prompt_message, tokenize=False, add_generation_prompt=True
                )
            except TemplateError:
                merged_user = f"[SYSTEM]\n{sys_prompt}\n\n[USER]\n{user_content}"
                prompt_message = [{"role": "user", "content": merged_user}]
                chat_text = tokenizer.apply_chat_template(
                    prompt_message, tokenize=False, add_generation_prompt=True
                )

            chat_prompts.append(chat_text)

        responses = model.generate(chat_prompts, sampling_params)
        judges =[]
        checks = []
        tokens = []
        for i, output in enumerate(responses):
            gens= [out.text.strip() for out in output.outputs]
            token = [len(out.token_ids) for out in output.outputs]
            check = check_correct(flips[i],gens)
            judges.append(gens)
            checks.append(check)
            tokens.append(token)
        dataset['judge'] = judges
        dataset['flip'] = flips
        dataset['check'] = checks
        dataset['num_tokens'] = tokens
        return dataset


def parse_binary_safe_judge(text: str) -> Optional[int]:
    """
    Rule:
    - contains only '1' → return 1
    - contains only '0' → return 0
    - contains both → return None
    - contains neither → return None
    """

    if text is None:
        return None

    s = str(text)

    has_1 = bool(re.search(r"\b1\b", s))
    has_0 = bool(re.search(r"\b0\b", s))

    if has_1 and not has_0:
        return 1
    if has_0 and not has_1:
        return 0

    return None


def check_safe_correct(gens):
    scores = []
    for gen in gens:
        parsed = parse_binary_safe_judge(gen)
        scores.append(0 if parsed is None else parsed)
    return scores


@torch.no_grad()
def generate_response_safe_vllm(
    dataset,
    model,
    model_name,
    tokenizer,
    sys_prompt,
    user_template,
    local_index,
    n=1,
    prompt_name="prompt",
    answer_name="response",
    seed=42,
):
    with torch.inference_mode():
        temp = 0
        max_tokens = 4096
        sampling_params = SamplingParams(
            temperature=temp,
            top_p=1.0,
            n=n,
            max_tokens=max_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
        )

        all_prompt = dataset[prompt_name].tolist()
        all_answer = dataset[answer_name].tolist()
        chat_prompts = []

        for i in range(len(all_prompt)):
            prompt = all_prompt[i]
            answer = all_answer[i]

            user_content = user_template.format(
                question=prompt,
                answer=answer,
            )

            try:
                prompt_message = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content},
                ]
                chat_text = tokenizer.apply_chat_template(
                    prompt_message,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TemplateError:
                merged_user = f"[SYSTEM]\n{sys_prompt}\n\n[USER]\n{user_content}"
                prompt_message = [{"role": "user", "content": merged_user}]
                chat_text = tokenizer.apply_chat_template(
                    prompt_message,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            chat_prompts.append(chat_text)

        responses = model.generate(chat_prompts, sampling_params)

        judges = []
        checks = []
        parsed_scores = []
        format_errors = []
        tokens = []

        for output in responses:
            gens = [out.text.strip() for out in output.outputs]
            token = [len(out.token_ids) for out in output.outputs]

            parsed = [parse_binary_safe_judge(g) for g in gens]
            check = [0 if p is None else p for p in parsed]
            fmt_err = [p is None for p in parsed]

            judges.append(gens)
            checks.append(check)
            parsed_scores.append(parsed)
            format_errors.append(fmt_err)
            tokens.append(token)

        dataset["judge"] = judges
        dataset["check"] = checks
        dataset["parsed_score"] = parsed_scores
        dataset["format_error"] = format_errors
        dataset["num_tokens"] = tokens

        return dataset
