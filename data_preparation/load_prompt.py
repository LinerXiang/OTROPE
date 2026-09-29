
def load_prompts(model_name_or_path):
    sys_prompt = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. "
    "You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider "
    "factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by "
    "comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were "
    "presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names "
    "of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "
    "[[A]] if assistant A is better, [[B]] if assistant B is better. Your final line must contain only one of these tokens."
)
    user_template = "[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"


    return sys_prompt, user_template

# actually, it's not related to model_name_or_path, all models apply the same prompt
def load_all_prompt(model_name_or_path):
    sys_prompt, user_template = load_prompts(model_name_or_path)
    return sys_prompt, user_template


def load_gen_prompts(model_name_or_path):
    sys_prompt = ("You are a helpful, clear, and accurate assistant. Provide the best possible answer to the user question below.")
    user_template = "[User Question]\n{question}\n\n"
    return sys_prompt, user_template



def load_all_gen_prompt(model_name_or_path):
    sys_prompt, user_template = load_gen_prompts(model_name_or_path)
    return sys_prompt, user_template


def load_gen_prompts_guide(model_name_or_path):
    sys_prompt = (
        "You are a careful and highly capable assistant."
            "Aim for a correct, complete, and well-justified answer."
            "Follow these principles:"
            "2. Prefer clear structure and precise wording."
            "Produce the best possible answer."
        )
    user_template = "[User Question]\n{question}\n\n"
    return sys_prompt, user_template



def load_all_gen_prompt_guide(model_name_or_path):
    sys_prompt, user_template = load_gen_prompts_guide(model_name_or_path)
    return sys_prompt, user_template
