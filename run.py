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

shortcuts = {
    "trivia": my_datasets.TRIVIA_SAMPLE_LOC.format(my_datasets.NUM_TO_KEEP),
    "hotpot": my_datasets.HOTPOT_SAMPLE_LOC.format(my_datasets.NUM_TO_KEEP),
    "llmqg": my_datasets.LLMQG_SAMPLE_LOC.format(my_datasets.NUM_TO_KEEP),
}
shortcuts["t"] = shortcuts["trivia"]
shortcuts["h"] = shortcuts["hotpot"]
shortcuts["l"] = shortcuts["llmqg"]


def gen_then_cache(src, f, cache_loc):
    if os.path.exists(cache_loc):
        with open(cache_loc, "rb") as f:
            return pickle.load(f)
    ret = process_map(f, src, max_workers=32)
    with open(cache_loc, "wb") as f:
        pickle.dump(ret, f)
    return ret


class CQA_Inspector:
    def stat(self, data_path):
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
        df = pd.DataFrame([len(i[1]) for i in cqas])
        print(df.describe())

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
        for l0, f0 in sorted(bow[0].items(), key=lambda x: -x[1]):
            if f0 < len(qs) * bar:
                break
            print(f"{l0}: {f0} ({f0/len(qs):.1%})")
            for l1, f1 in sorted(bow[1].items(), key=lambda x: -x[1]):
                if f1 < len(qs) * bar:
                    break
                if l1[:1] == l0:
                    print(f"\t{l1}: {f1} ({f1/len(qs):.1%})")
                    for l2, f2 in sorted(bow[2].items(), key=lambda x: -x[1]):
                        if f2 < len(qs) * bar:
                            break
                        if l2[:2] == l1:
                            print(f"\t\t{l2}: {f2} ({f2/len(qs):.1%})")

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
        so_far = 0
        for i in range(1, 11):
            print(f"Type {i}: {cnt[i]} ({cnt[i]/len(qs):.1%}) | {qts[i-1]}")
            so_far += cnt[i]
        print(f"Others: {len(qs)-so_far} ({(len(qs)-so_far)/len(qs):.1%}) | Others")


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
        for k, v in sorted(cnt.items(), reverse=True):
            print(f"{k}: {v} ({v/len(star):.1%})")

    def len_req(self, data_path, gen_path=None, stat_only=False):
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
            # debug = sorted([(llm_utils.word_cnt(i[2]), i[2]) for i in cqas])
            # print(debug[-5:])
            print(df.describe())
            return
        
        if gen_path is None:
            gen_path = data_path.replace("cqas", "min_ans_len")
            
            ans = gen_then_cache(
                cqas,
                llm_utils.generate_ans,
                data_path.replace("cqas", f"gen_a_wc"),
            )
            star = gen_then_cache(
                [(x, y) for x, y in zip(cqas, ans)],
                llm_utils.check_ans_star,
                data_path.replace("cqas", f"gen_a_wc_star"),
            )
            shorter = gen_then_cache(
                [(x, a, r) for (x, a, r) in zip(cqas, ans, star)],
                packed_detect_ans_len_req,
                gen_path,
            )  # return (word count, trials)
            # qtype = gen_then_cache(
            #     [x[1] for x in cqas],
            #     llm_utils.classify_question_type,
            #     data_path.replace("cqas", "qtype"),
            # )

            print(f"# Minimize answer length - {data_path}")
            df = pd.DataFrame([x for x, _ in shorter])
            print(df.describe())
            # df = pd.DataFrame([x for (x, _), t in zip(shorter, qtype) if t[0] == 2])
            # print(df.describe())

    def coverage(self, data_path, gen_path):
        pass  # TODO


if __name__ == "__main__":
    fire.Fire(CQA_Inspector)
