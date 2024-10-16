import pickle
import fire
import my_datasets
import pandas as pd
from collections import Counter
import os
from tqdm.contrib.concurrent import process_map
import llm_utils
import prompts
from ans_len import packed_detect_ans_len_req
from coverage import detect_coverage

shortcuts = {
    "trivia": my_datasets.TRIVIA_SAMPLE_LOC.format(my_datasets.NUM_TO_KEEP),
    "hotpot": my_datasets.HOTPOT_SAMPLE_LOC.format(my_datasets.NUM_TO_KEEP),
    "llmqg": my_datasets.LLMQG_GPT_SAMPLE_LOC.format(my_datasets.NUM_TO_KEEP),
    "llmqg_gpt": my_datasets.LLMQG_GPT_SAMPLE_LOC.format(my_datasets.NUM_TO_KEEP),
    "llmqg_llama": my_datasets.LLMQG_LLAMA_SAMPLE_LOC.format(my_datasets.NUM_TO_KEEP),
}
shortcuts["t"] = shortcuts["trivia"]
shortcuts["h"] = shortcuts["hotpot"]
shortcuts["l"] = shortcuts["llmqg"]
shortcuts["lg"] = shortcuts["llmqg_gpt"]
shortcuts["ll"] = shortcuts["llmqg_llama"]

question_types = {
    1: "B",
    2: "A",
    3: "A",
    4: "A",
    5: "C",
    6: "C",
    7: "A",
    8: "C",
    9: "B",
    10: "A"
}


def gen_then_cache(src, f, cache_loc):
    if os.path.exists(cache_loc):
        with open(cache_loc, "rb") as f:
            return pickle.load(f)
    ret = process_map(f, src, max_workers=32)
    with open(cache_loc, "wb") as f:
        pickle.dump(ret, f)
    return ret


class CQA_Inspector:
    def stat(self, data_path, group=None):
        if data_path in shortcuts:
            data_path = shortcuts[data_path]
        print(f"# Stat - {data_path}")
        with open(data_path, "rb") as f:
            cqas = pickle.load(f)
        print(
            "# Sample num:",
            len(cqas),
        )
        print("# Question Length stat")
        df = pd.DataFrame([len(i[1].split(" ")) for i in cqas])
        if group is not None:
            if group not in {"A", "B", "C"}:
                raise ValueError("Group must be one of 'A', 'B', or 'C'.")
            output_to = data_path.replace("cqas", "qtype")
            qs = [x[1] for x in cqas]
            qtype = gen_then_cache(
                qs,
                llm_utils.classify_question_type,
                output_to,
            )
            qtype_df = pd.DataFrame(qtype, columns=["question_type", "question_description"])
            cqa_df = pd.DataFrame(cqas, columns=["context", "question", "answer"])
            if len(qtype_df) != len(cqa_df):
                raise ValueError("Mismatch between number of cqas and qtype entries.")
            merged_df = pd.concat([cqa_df, qtype_df], axis=1)
            merged_df["group"] = merged_df["question_type"].map(question_types)
            unknown_count = merged_df["group"].isna().sum()
            if unknown_count > 0:
                print(f"# Warning: {unknown_count} questions have undefined groups and will be excluded.")
                merged_df = merged_df.dropna(subset=["group"])
            if group is not None:
                if group not in {"A", "B", "C"}:
                    raise ValueError("Group must be one of 'A', 'B', or 'C'.")
                filtered_df = merged_df[merged_df["group"] == group]
                print("# Filtered Sample num:", len(filtered_df))
                print("# Group Ratio:", len(filtered_df) / len(merged_df))
            else:
                filtered_df = merged_df
            df = pd.DataFrame([len(i.split(" ")) for i in filtered_df["question"]])

        stats = df.describe()
        print(stats)
        print()
        print("Returned dict: ")
        return stats[0].to_dict()

    def start_word(self, data_path, bar=0.01):
        if data_path in shortcuts:
            data_path = shortcuts[data_path]
        print(f"# Words at the beginning - {data_path}")
        with open(data_path, "rb") as f:
            cqas = pickle.load(f)

        qs = [x[1] for x in cqas]
        bow = [Counter(), Counter(), Counter()]
        for q in qs:
            ws = q.split()
            while len(ws) < 3:
                ws.append("")
            for i in range(3):
                bow[i][tuple(ws[: i + 1])] += 1
        results = {1: {}, 2: {}, 3: {}}
        for l0, f0 in sorted(bow[0].items(), key=lambda x: -x[1]):
            if f0 < len(qs) * bar:
                break
            print(f"{l0}: {f0} ({f0/len(qs):.1%})")
            results[1][l0] = f0 / len(qs)
            for l1, f1 in sorted(bow[1].items(), key=lambda x: -x[1]):
                if f1 < len(qs) * bar:
                    break
                if l1[:1] == l0:
                    print(f"\t{l1}: {f1} ({f1/len(qs):.1%})")
                    results[2][l1] = f1 / len(qs)
                    for l2, f2 in sorted(bow[2].items(), key=lambda x: -x[1]):
                        if f2 < len(qs) * bar:
                            break
                        if l2[:2] == l1:
                            print(f"\t\t{l2}: {f2} ({f2/len(qs):.1%})")
                            results[3][l2] = f2 / len(qs)
        return results

    def qtype(self, data_path, output_to=None):
        if data_path in shortcuts:
            data_path = shortcuts[data_path]
        print(f"# Question Type - {data_path}")
        with open(data_path, "rb") as f:
            cqas = pickle.load(f)

        if output_to is None:
            output_to = data_path.replace("cqas", "qtype")

        qs = [x[1] for x in cqas]
        qtype = gen_then_cache(
            qs,
            llm_utils.classify_question_type,
            output_to,
        )
        cnt = Counter([x[0] for x in qtype])
        qts = [qt[:128] + "..." for qt in prompts.QUESTION_TYPES.split("\n")]
        results = {}
        so_far = 0
        for i in range(1, 11):
            count = cnt[i]
            percentage = count / len(qs)
            description = qts[i - 1]
            print(f"Type {i}: {count} ({percentage:.1%}) | {description}")
            results[f"Type {i}"] = {"count": count, "percentage": percentage, "description": description}
            so_far += count
        others_count = len(qs) - so_far
        others_percentage = others_count / len(qs)
        print(f"Others: {others_count} ({others_percentage:.1%}) | Others")
        results["Others"] = {"count": others_count, "percentage": others_percentage, "description": "Others"}
        return results

    def answerable(self, data_path, gen_ans=None, judge_ans=None, use_ctx=True):
        if data_path in shortcuts:
            data_path = shortcuts[data_path]
        if use_ctx:
            print(f"# Answerable - {data_path}")
        else:
            print(f"# Uncommonness - {data_path}")
        with open(data_path, "rb") as f:
            cqas = pickle.load(f)

        tag = "wc" if use_ctx else "woc"
        if gen_ans is None:
            gen_ans = data_path.replace("cqas", f"gen_a_{tag}")
        if judge_ans is None:
            judge_ans = data_path.replace("cqas", f"gen_a_{tag}_star")

        ans = gen_then_cache(
            (cqas if use_ctx else [(None, x[1], x[2]) for x in cqas]),
            llm_utils.generate_ans,
            gen_ans,
        )
        star = gen_then_cache(
            [(x, y) for x, y in zip(cqas, ans)],
            llm_utils.check_ans_star,
            judge_ans,
        )
        cnt = Counter([x[0] if x[0] is not None else -1 for x in star])
        results = {}
        for k, v in sorted(cnt.items(), reverse=True):
            percentage = v / len(star)
            label = f"{k}" if k != -1 else "Undefined"
            print(f"{label}: {v} ({percentage:.1%})")
            results[label] = {"count": v, "percentage": percentage}
        return results

    def len_req(self, data_path, gen_path=None, stat_only=False, group=None):
        if data_path in shortcuts:
            data_path = shortcuts[data_path]
        with open(data_path, "rb") as f:
            cqas = pickle.load(f)

        if cqas[0][2] is not None:  # golden ans exists, just output stats
            stat_only = True
        if stat_only:
            if "llmqg" in data_path:
                if not os.path.exists(data_path.replace("cqas", "gen_a_wc")):
                    print("Please generate answers with ctx first.")
                    return
                ans = gen_then_cache(
                    cqas,
                    llm_utils.generate_ans,
                    data_path.replace("cqas", "gen_a_wc"),
                )
                cqas = [(x[0], x[1], y) for x, y in zip(cqas, ans)]
            print(f"# Answer word cnt stat - {data_path}")
            df = pd.DataFrame([llm_utils.word_cnt(i[2]) for i in cqas])

            if group is not None:
                if group not in {"A", "B", "C"}:
                    raise ValueError("Group must be one of 'A', 'B', or 'C'.")
                output_to = data_path.replace("cqas", "qtype")
                qs = [x[1] for x in cqas]
                qtype = gen_then_cache(
                    qs,
                    llm_utils.classify_question_type,
                    output_to,
                )
                qtype_df = pd.DataFrame(qtype, columns=["question_type", "question_description"])
                if len(qtype_df) != len(df):
                    raise ValueError("Mismatch between number of cqas and qtype entries.")
                merged_df = pd.concat([df, qtype_df], axis=1)
                merged_df["group"] = merged_df["question_type"].map(question_types)
                unknown_count = merged_df["group"].isna().sum()
                if unknown_count > 0:
                    print(f"# Warning: {unknown_count} questions have undefined groups and will be excluded.")
                    merged_df = merged_df.dropna(subset=["group"])
                filtered_df = merged_df[merged_df["group"] == group]
                print("# Filtered Sample num:", len(filtered_df))
                print("# Group Ratio:", len(filtered_df) / len(merged_df))
                df = filtered_df

            stats = df.describe()
            print(stats)
            return {
                "minimize_answer_length_stats": stats[0].to_dict(),
                "reduction_rate_stats": stats[0].to_dict(),
            }

        if gen_path is None:
            gen_path = data_path.replace("cqas", "min_ans_len")

        ans = gen_then_cache(
            cqas,
            llm_utils.generate_ans,
            data_path.replace("cqas", "gen_a_wc"),
        )
        star = gen_then_cache(
            [(x, y) for x, y in zip(cqas, ans)],
            llm_utils.check_ans_star,
            data_path.replace("cqas", "gen_a_wc_star"),
        )
        shorter = gen_then_cache(
            [(x, a, r) for (x, a, r) in zip(cqas, ans, star)],
            packed_detect_ans_len_req,
            gen_path,
        )

        print(f"# Minimize answer length - {data_path}")
        df = pd.DataFrame([x for x, _ in shorter])

        if group is not None:
            if group not in {"A", "B", "C"}:
                raise ValueError("Group must be one of 'A', 'B', or 'C'.")
            output_to = data_path.replace("cqas", "qtype")
            qs = [x[1] for x in cqas]
            qtype = gen_then_cache(
                qs,
                llm_utils.classify_question_type,
                output_to,
            )
            qtype_df = pd.DataFrame(qtype, columns=["question_type", "question_description"])
            if len(qtype_df) != len(df):
                raise ValueError("Mismatch between number of cqas and qtype entries.")
            merged_df = pd.concat([df, qtype_df], axis=1)
            merged_df["group"] = merged_df["question_type"].map(question_types)
            unknown_count = merged_df["group"].isna().sum()
            if unknown_count > 0:
                print(f"# Warning: {unknown_count} questions have undefined groups and will be excluded.")
                merged_df = merged_df.dropna(subset=["group"])
            filtered_df = merged_df[merged_df["group"] == group]
            print("# Filtered Sample num:", len(filtered_df))
            print("# Group Ratio:", len(filtered_df) / len(merged_df))
            df = filtered_df

        print(df.describe())
        print("## Reduction rate:")
        df_reduction = pd.DataFrame([x / llm_utils.word_cnt(a) for (x, _), a in zip(shorter, ans)])
        print(df_reduction.describe())
        return {
            "minimize_answer_length_stats": df.describe()[0].to_dict(),
            "reduction_rate_stats": df_reduction.describe()[0].to_dict(),
        }

    def cover(self, data_path, gen_path=None):
        if data_path in shortcuts:
            data_path = shortcuts[data_path]
        print(f"# Coverage - {data_path}")
        with open(data_path, "rb") as f:
            cqas = pickle.load(f)

        if gen_path is None:
            gen_path = data_path.replace("cqas", "cov")

        cov = gen_then_cache(
            cqas,
            detect_coverage,
            gen_path,
        )
        print("## word level")
        df_word_level = pd.DataFrame([x for x, _, _ in cov])
        print(df_word_level.describe())
        word_level_stats = df_word_level.describe().to_dict()

        print("## word cnt")
        df_word_cnt = pd.DataFrame([x['total'] for _, _, x in cov])
        print(df_word_cnt.describe())
        word_cnt_stats = df_word_cnt.describe().to_dict()

        print("## sent level")
        df_sent_level = pd.DataFrame([x for _, x, _ in cov])
        print(df_sent_level.describe())
        sent_level_stats = df_sent_level.describe().to_dict()

        print("## sent cnt")
        df_sent_cnt = pd.DataFrame([len(x['sents']) for _, _, x in cov])
        print(df_sent_cnt.describe())
        sent_cnt_stats = df_sent_cnt.describe().to_dict()

        return {
            "word_level_stats": word_level_stats,
            "word_cnt_stats": word_cnt_stats,
            "sent_level_stats": sent_level_stats,
            "sent_cnt_stats": sent_cnt_stats,
        }


if __name__ == "__main__":
    fire.Fire(CQA_Inspector)