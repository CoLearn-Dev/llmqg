import json
import os
import pickle
import pandas as pd
from typing import List, Tuple, Optional
from llm_utils import generate_wiki_question, retry_until
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import random


Context = Optional[str]
Question = Optional[str]
Answer = Optional[str]
CQ = Tuple[Context, Question]
CQS = Tuple[Context, Question]
CQAS = Tuple[Context, Question, Answer]

NUM_TO_KEEP = 256

TRIVIA_DEFAULT_LOC = "./data/triviaqa-unfiltered"
TRIVIA_SAMPLE_LOC = "./data/trivia.cqas{}.pkl"
HOTPOT_DEFAULT_LOC = "./data/hotpotqa-fullwiki"
HOTPOT_SAMPLE_LOC = "./data/hotpot.cqas{}.pkl"
WIKI_CSV_LOC = "./data/wiki_text_cleaned_v1.csv"
LLMQG_SAMPLE_LOC = "./data/llmqg.cqas{}.pkl"


def load_trivia_cqas(loc=TRIVIA_DEFAULT_LOC):
    # note that trivia does not provide context
    TRIVIA_TRAIN_FILENAME = "unfiltered-web-train.json"
    TRIVIA_DEV_FILENAME = "unfiltered-web-dev.json"
    TRIVIA_TEST_FILENAME = "unfiltered-web-test-without-answers.json"
    with open(os.path.join(loc, TRIVIA_TRAIN_FILENAME), "r") as fin:
        d_train = json.load(fin)
    # >>> d_train.keys()
    # dict_keys(['Data', 'Domain', 'Split', 'VerifiedEval', 'Version'])
    # >>> d_train['Data'][0].keys()
    # dict_keys(['Answer', 'EntityPages', 'Question', 'QuestionId', 'QuestionSource', 'SearchResults'])
    cqas = []
    for d in d_train["Data"]:
        cqa = (None, d["Question"], d["Answer"]["Value"])
        cqas.append(cqa)
    return cqas


def dump_trivia_samples(n=NUM_TO_KEEP):
    with open(TRIVIA_SAMPLE_LOC.format(n), "wb") as f:
        pickle.dump(random.sample(load_trivia_cqas(), n), f)


def load_trivia_samples(n=NUM_TO_KEEP):
    with open(TRIVIA_SAMPLE_LOC.format(n), "rb") as f:
        return pickle.load(f)


def extract_hotpot_cqas(instance):
    q = instance["question"]
    a = instance["answer"]
    selected_c = {}
    for n, i in instance["supporting_facts"]:
        if n in selected_c:
            selected_c[n].add(i)
        else:
            selected_c[n] = {i}
    c = ""
    for cn, cl in instance["context"]:
        if cn in selected_c:
            c += f"{cn}\n"
            # for i in sorted(selected_c[cn]):
            #     if i >= len(cl):
            #         break  # toxic dataset... bug sometimes
            #     c += f"{cl[i]}\n"
            # apparently the labeling for index is not precise, just load the full ctx
            for l in cl:
                c += f"{l}\n"
            c += "\n"
    return (c, q, a)


def load_hotpot_cqas(loc=HOTPOT_DEFAULT_LOC):
    HOTPOT_TRAIN_FILENAME = "hotpot_train_v1.1.json"
    with open(os.path.join(loc, HOTPOT_TRAIN_FILENAME), "r") as fin:
        d_train = json.load(fin)
    cqas = []
    for instance in d_train:
        cqas.append(extract_hotpot_cqas(instance))
    return cqas


def dump_hotpot_samples(n=NUM_TO_KEEP):
    with open(HOTPOT_SAMPLE_LOC.format(n), "wb") as f:
        pickle.dump(random.sample(load_hotpot_cqas(), n), f)


def load_hotpot_samples(n=NUM_TO_KEEP):
    with open(HOTPOT_SAMPLE_LOC.format(n), "rb") as f:
        return pickle.load(f)


def wiki_to_ctx(x):
    page_name = x[1]
    section_name = x[2] if x[2] != "nan" else None
    sub_section_name = x[3] if x[3] != "nan" else None
    sub_sub_section_name = x[4] if x[4] != "nan" else None
    text = x[5]
    ctx = ""
    # In an article about '{paragraph.page_name}', section '{paragraph.section_name}', subsection '{paragraph.subsection_name}', paragraph '{paragraph.subsubsection_name}'
    # mentioned: \n {paragraph.text_cleaned}
    if page_name:
        ctx += f"In an article about '{page_name}'"
    if section_name:
        ctx += f", section '{section_name}'"
    if sub_section_name:
        ctx += f", subsection '{sub_section_name}'"
    if sub_sub_section_name:
        ctx += f", paragraph '{sub_sub_section_name}'"
    if text:
        if ctx:
            ctx += f", it mentioned: \n"
        ctx += f"{text}"
    return ctx


def c_to_cqs(p):
    c, qa_per_ctx = p
    qs = retry_until(
        generate_wiki_question,
        {"ctx": c, "num_questions": qa_per_ctx},
        lambda x: len(x) == qa_per_ctx,
    )
    return [(c, q, None) for q in qs]


def generate_llmqg_samples(n=NUM_TO_KEEP, qa_per_ctx=1):
    # return n/qa_per_ctx instances, each with qa_per_ctx questions
    assert n % qa_per_ctx == 0
    # 1. filter input set, sample required number of contexts
    wiki_df = pd.read_csv(WIKI_CSV_LOC)
    wiki_df = wiki_df[
        wiki_df.word_count > 64
    ]  # filter out short wiki that does not have sufficient information for generation
    wiki_df = wiki_df[wiki_df.is_bad == 0]  # filter out bad wiki
    assert len(wiki_df) >= n / qa_per_ctx
    wiki_df = wiki_df.sample(n // qa_per_ctx)
    cs = []
    for x in wiki_df.values:
        cs.append((wiki_to_ctx(x), qa_per_ctx))
    # 2. invoke generation and organize into the required format
    cqs = process_map(
        c_to_cqs,
        cs,
        max_workers=8,
    )
    return sum(cqs, [])


def dump_llmqg_samples(n=NUM_TO_KEEP, qa_per_ctx=4):
    with open(LLMQG_SAMPLE_LOC.format(n), "wb") as f:
        pickle.dump(generate_llmqg_samples(n, qa_per_ctx), f)


def load_llmqg_samples(n=NUM_TO_KEEP):
    with open(LLMQG_SAMPLE_LOC.format(n), "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    dump_trivia_samples()
    dump_hotpot_samples()
    dump_llmqg_samples()
