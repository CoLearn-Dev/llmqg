import os
import openai
import re
import random
from together import Together
from prompts import (
    QUESTION_GENERATION_SYS_PROMPT,
    SUMMARIZE_QUESTION_TYPE_SYS_PROMPT,
    CLASSIFY_QUESTION_TYPE_SYS_PROMPT,
    GENERATE_ANS_SYS_PROMPT,
    GENERATE_ANS_SHORT_SYS_PROMPT,
    GENERATE_LIMIT_NUM_ANS_SYS_PROMPT,
    CHECK_ANS_STAR_SYS_PROMPT,
    SELECT_RELEVANT_SENTS_SYS_PROMPT
)
from my_config import OPENAI_API_KEY, TOGETHER_API_KEY

openai_client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
)
together_client = Together(
    api_key=TOGETHER_API_KEY
)

def word_cnt(s):
    return len([x.strip() for x in s.split() if len(x.strip()) > 1])

def retry_until(f, kargs, p, retry=3):
    for i in range(retry):
        try:
            r = f(**kargs)
            if p(r):
                return r
        except Exception as e:
            print("Error:", e)
            pass
    print("Failed after retrying")
    return p

def get_completion(model, messages, **kwargs):
    if model.startswith("gpt-"):
        try:
            completion = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=kwargs.get("temperature", 0),
                **kwargs
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI completion error: {e}")
            raise
    else:
        try:
            response = together_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=kwargs.get("temperature", 0),
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Together AI completion error: {e}")
            raise

def generate_wiki_question(ctx, num_questions=1, model="gpt-4o"):
    completion_text = get_completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": QUESTION_GENERATION_SYS_PROMPT.format(
                    NUM_QUESTIONS=num_questions
                ),
            },
            {
                "role": "user",
                "content": ctx,
            },
        ],
    )
    generated_text = completion_text.strip()
    new_questions = [
        re.sub(r"^\d\.", "", x).strip()
        for x in generated_text.split("\n")
        if re.match(r"^[0-9]\.", x)
    ]
    if len(new_questions) >= num_questions:
        return new_questions[:num_questions]
    else:
        print("Warning: Not enough questions generated. ctx:", ctx)
        return new_questions

def summarize_question_types(qs, model="gpt-4o"):
    completion_text = get_completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SUMMARIZE_QUESTION_TYPE_SYS_PROMPT,
            },
            {
                "role": "user",
                "content": "\n".join(
                    [f"{i+1}. {q}" for i, q in enumerate(random.sample(qs, min(len(qs), 128)))]
                ),
            },
        ],
    )
    print(completion_text)

def classify_question_type(q, model="gpt-4o"):
    completion_text = get_completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": CLASSIFY_QUESTION_TYPE_SYS_PROMPT,
            },
            {
                "role": "user",
                "content": q,
            },
        ],
    )
    generated = completion_text
    lines = generated.strip().split("\n")

    def try_parse_int(x):
        try:
            r = int(x)
            if r < 1 or r > 10:
                return None
            return r
        except Exception as e:
            print(e)
            return None

    if len(lines) == 0:
        return None, None
    if len(lines) == 1:
        return try_parse_int(lines[0].strip()), None
    return try_parse_int(lines[0].strip()), lines[1].strip()

def generate_ans(x, enforce_short=None, model="gpt-4o"):
    (c, q, _a) = x
    system_prompt = (
        GENERATE_ANS_SYS_PROMPT
        if enforce_short is None
        else (
            GENERATE_ANS_SHORT_SYS_PROMPT
            if enforce_short == 0
            else GENERATE_LIMIT_NUM_ANS_SYS_PROMPT.format(enforce_short)
        )
    )
    user_content = f"Question:\n{q}"
    if c:
        user_content += f"\nSupporting fact:\n{c}"
    completion_text = get_completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
    )
    a = completion_text.strip()
    return a

def check_ans_star(p, model="gpt-4o"):
    (c, q, a), aa = p
    completion_text = get_completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": CHECK_ANS_STAR_SYS_PROMPT,
            },
            {
                "role": "user",
                "content": f"Question:\n{q}\nAnswer:\n{aa}"
                + (f"\nSupporting fact:\n{c}" if c else "")
                + (f"\nSupporting fact:\n{a}" if a else ""),
            },
        ],
    )
    generated = completion_text
    lines = generated.strip().split("\n")

    def try_parse_int(x):
        try:
            r = int(x[0])
            if r < 0 or r > 5:
                return -1
            return r
        except Exception as e:
            print("Error:", e)
            return -1

    if len(lines) == 0:
        return -1, ""
    if len(lines) == 1:
        return try_parse_int(lines[0].strip()), ""
    return try_parse_int(lines[0].strip()), lines[1].strip()

def select_relevant_sents(q, sents, model="gpt-4o"):
    sent_list = "\n".join([f"{i+1}. {s}" for i, s in enumerate(sents)])
    completion_text = get_completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SELECT_RELEVANT_SENTS_SYS_PROMPT,
            },
            {
                "role": "user",
                "content": f"Question:\n{q}\nSentences:\n{sent_list}",
            },
        ],
    )
    generated = completion_text

    def try_parse_int(x):
        try:
            r = int(x)
            if r < 1 or r > len(sents):
                return -1
            return r
        except Exception as e:
            print("Error:", e)
            return -1

    sent_ids = [try_parse_int(t.strip()) for t in generated.split(",")]
    sent_ids = set([x - 1 for x in sent_ids if x != -1])  # filter out -1
    return sent_ids

if __name__ == "__main__":
    print(
        generate_wiki_question(
            "In an article about 'Trans @-@ Alaska Pipeline System', section 'Additional sources', it mentioned: Fineberg , Richard A. A Pipeline in Peril : A Status Report on the Trans-Alaska Pipeline . Ester , Alaska ; Alaska Forum for Environmental Responsibility , 1996 ."
        )
    )
    print(
        generate_wiki_question(
            "In an article about 'Trans @-@ Alaska Pipeline System', section 'Additional sources', it mentioned: Fineberg , Richard A. A Pipeline in Peril : A Status Report on the Trans-Alaska Pipeline . Ester , Alaska ; Alaska Forum for Environmental Responsibility , 1996 .",
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        )
    )