import pandas as pd
import ast
import os
import time
import logging
import random
from utils.llm_utils import get_completion
from dotenv import load_dotenv
from tqdm import tqdm
from scipy.stats import pearsonr

load_dotenv()

# Load the master dataframe
df_10k = pd.read_csv("data/askme-qa/raw_data_10k.csv")

# Directory structure
base_dir = "data/hitl"
annotators = ["HITL_Alex", "HITL_Atheer", "HITL_Hend", "HITL_Xiaoyuan"]
file_names = [
    "Copy of answers_samples.xlsx",
    "Copy of questions_samples.xlsx",
    "Copy of questions_zs_samples.xlsx",
]

MODEL = "gpt-4o"


def rate_answer_with_fact(question, answer, fact):
    if not str(question).strip() or not str(answer).strip() or not str(fact).strip():
        logging.debug(
            "Missing input in rate_answer_with_fact: %s, %s, %s", question, answer, fact
        )
        return {"score": None, "rationale": "Missing input."}
    time.sleep(random.uniform(0.5, 1.5))  # avoid rate limits
    prompt_template = (
        "Based on this fact: \n\n `{REFERENCE}` \n\n "
        "Rate the following answer to the question - Question: `{QUESTION}` \n\n "
        "Answer: `{ANSWER}`; give a number from 0-5 where 0 is 'No answer or completely irrelevant', "
        "1 is 'Significantly incorrect or incomplete', 2 is 'Partially correct; major inaccuracies or omissions', "
        "3 is 'Correct but lacks depth; minimal detail', 4 is 'Mostly correct; minor errors, includes relevant details', "
        "5 is 'Fully accurate and detailed; clear and comprehensive'. Your answer should follow the form "
        "`Answer:<number> \n Rationale:<justify your judgment in a paragraph>`."
    )
    prompt = prompt_template.format(REFERENCE=fact, QUESTION=question, ANSWER=answer)
    output = get_completion(MODEL, [{"role": "user", "content": prompt}])
    # Parse output
    import re

    score_match = re.search(r"Answer\s*:\s*(\d)", output)
    rationale_match = re.search(r"Rationale\s*:\s*(.*)", output, re.DOTALL)
    score = int(score_match.group(1)) if score_match else None
    rationale = rationale_match.group(1).strip() if rationale_match else output.strip()
    return {"score": score, "rationale": rationale}


# Function to calculate average correlation between human annotators
def average_human_correlation(df):
    annotator_columns = [
        col
        for col in df.columns
        if col.startswith("annotation_") and col != "annotation_llm"
    ]
    n = len(annotator_columns)
    correlations = []

    for i in range(n):
        for j in range(i + 1, n):
            common_data = df[[annotator_columns[i], annotator_columns[j]]].dropna()
            if (
                not common_data.empty
                and common_data[annotator_columns[i]].nunique() > 1
                and common_data[annotator_columns[j]].nunique() > 1
            ):
                corr, _ = pearsonr(
                    common_data[annotator_columns[i]], common_data[annotator_columns[j]]
                )
                correlations.append(corr)
            elif (
                common_data[annotator_columns[i]].nunique() == 1
                and common_data[annotator_columns[j]].nunique() == 1
            ):
                if (
                    common_data[annotator_columns[i]].unique()[0]
                    == common_data[annotator_columns[j]].unique()[0]
                ):
                    correlations.append(1)
                else:
                    correlations.append(-1)

    return sum(correlations) / len(correlations) if correlations else 0


# Function to calculate average correlation between human annotators and machine annotator
def average_human_machine_correlation(df):
    human_columns = [
        col
        for col in df.columns
        if col.startswith("annotation_") and col != "annotation_llm"
    ]

    df["annotation_score"] = df["annotation_llm"].apply(
        lambda x: ast.literal_eval(x)["score"]
    )

    machine_column = "annotation_score"
    correlations = []

    for human_col in human_columns:
        common_data = df[[human_col, machine_column]].dropna()
        if (
            not common_data.empty
            and common_data[human_col].nunique() > 1
            and common_data[machine_column].nunique() > 1
        ):
            corr, _ = pearsonr(common_data[human_col], common_data[machine_column])
            correlations.append(corr)
        elif (
            common_data[human_col].nunique() == 1
            and common_data[machine_column].nunique() == 1
        ):
            if (
                common_data[human_col].unique()[0]
                == common_data[machine_column].unique()[0]
            ):
                correlations.append(1)
            else:
                correlations.append(-1)

    return sum(correlations) / len(correlations) if correlations else 0


if os.path.exists("data/hitl/answers.csv"):
    df_answers = pd.read_csv("data/hitl/answers.csv")
else:
    # Initialize empty DataFrames for each type of samples
    df_answers = None
    df_questions = None
    df_questions_zs = None

    # Iterate over each annotator and each file
    for annotator in annotators:
        for file_name in file_names:
            file_path = os.path.join(base_dir, annotator, file_name)
            df = pd.read_excel(file_path)

            # Drop "instructions" column and unnamed columns
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
            if "instructions" in df.columns:
                df = df.drop(columns=["instructions"])
            if "Comments" in df.columns:
                df = df.drop(columns=["Comments"])
            # Drop columns with names that are not in English
            df = df.drop(columns=[col for col in df.columns if not col.isascii()])

            # Rename "annotation" column to include the annotator's name
            if "annotation" in df.columns:
                annotator_name = annotator.split("_")[-1].lower()
                df = df.rename(columns={"annotation": f"annotation_{annotator_name}"})
                annotation_col = df[
                    [f"annotation_{annotator_name}"]
                ]  # Select only the renamed annotation column

                # Merge data to respective DataFrames horizontally
                if "answers" in file_name:
                    if df_answers is None:
                        df_answers = df[["answer_id"]].join(annotation_col)
                    else:
                        df_answers = df_answers.join(
                            annotation_col, rsuffix=f"_{annotator_name}"
                        )

    df_10k_ic = df_10k[df_10k["setting"] == "ic"]
    df_10k_zs = df_10k[df_10k["setting"] == "zs"]
    df_answers = df_answers.merge(
        df_10k[
            ["id_answer", "value", "context", "text_paragraph", "text_question", "text"]
        ],
        left_on="answer_id",
        right_on="id_answer",
        how="left",
    ).drop(columns=["id_answer"])

    print(df_answers.to_dict("records")[0])

    df_answers["annotation_llm"] = [
        rate_answer_with_fact(
            row["text_question"],
            row["text"],
            row["context"] + " mentioned: " + row["text_paragraph"],
        )
        for row in tqdm(df_answers.to_dict("records"))
    ]
    for col in df_answers.columns:
        if col.startswith("annotation_"):
            df_answers[col] = df_answers[col].replace({"Y": True, "N": False})

    df_answers.to_csv("data/hitl/answers.csv", index=False)


# Calculate correlations for each type of sample
avg_human_corr_answers = average_human_correlation(df_answers)
avg_human_machine_corr_answers = average_human_machine_correlation(df_answers)

# Print the results
print("Average Human Correlation for Answers:", avg_human_corr_answers)
print("Average Human-Machine Correlation for Answers:", avg_human_machine_corr_answers)
