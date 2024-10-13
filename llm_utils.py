import openai
import re
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
import random
from my_config import OPENAI_API_KEY

client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
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


def generate_wiki_question(ctx, num_questions=1):
    completion = client.chat.completions.create(
        model="gpt-4o",
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
    generated_text = completion.choices[0].message.content.strip()
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


def summarize_question_types(qs):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": SUMMARIZE_QUESTION_TYPE_SYS_PROMPT,
            },
            {
                "role": "user",
                "content": "\n".join(
                    [f"{i+1}. {q}" for i, q in enumerate(random.sample(qs, 128))]
                ),
            },
        ],
    )
    print(completion.choices[0].message.content)


def classify_question_type(q):
    completion = client.chat.completions.create(
        model="gpt-4o",
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
    generated = completion.choices[0].message.content
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


def generate_ans(x, enforce_short=None):
    (c, q, _a) = x
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    GENERATE_ANS_SYS_PROMPT
                    if enforce_short is None
                    else (
                        GENERATE_ANS_SHORT_SYS_PROMPT
                        if enforce_short == 0
                        else GENERATE_LIMIT_NUM_ANS_SYS_PROMPT.format(enforce_short)
                    )
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{q}"
                + (f"\nSupporting fact:\n{c}" if c else ""),
            },
        ],
    )
    a = completion.choices[0].message.content.strip()
    return a


def check_ans_star(p):
    (c, q, a), aa = p
    completion = client.chat.completions.create(
        model="gpt-4o",
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
    generated = completion.choices[0].message.content
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


def select_relevant_sents(q, sents):
    sent_list = "\n".join([f"{i+1}. {s}" for i, s in enumerate(sents)])
    completion = client.chat.completions.create(
        model="gpt-4o",
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
    generated = completion.choices[0].message.content

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
    # print(generated, sent_ids)
    return sent_ids


if __name__ == "__main__":
    print(
        generate_wiki_question(
            "In an article about 'Trans @-@ Alaska Pipeline System', section 'Additional sources', it mentioned: Fineberg , Richard A. A Pipeline in Peril : A Status Report on the Trans-Alaska Pipeline . Ester , Alaska ; Alaska Forum for Environmental Responsibility , 1996 ."
        )
    )
