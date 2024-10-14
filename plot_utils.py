import os
import pickle
import fire
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from run import CQA_Inspector, shortcuts

class PlotUtils:
    def __init__(self, output_dir="plots"):
        self.datasets = ["trivia", "hotpot", "llmqg"]
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.inspector = CQA_Inspector()

    def _load_cqas(self, dataset):
        data_path = shortcuts.get(dataset, dataset)
        with open(data_path, "rb") as f:
            return pickle.load(f)

    def plot_stat(self):
        sample_nums = {}
        q_length_means = {}
        q_length_stds = {}

        for ds in self.datasets:
            cqas = self._load_cqas(ds)
            sample_nums[ds] = len(cqas)
            q_lengths = [len(q[1]) for q in cqas]
            q_length_means[ds] = pd.Series(q_lengths).mean()
            q_length_stds[ds] = pd.Series(q_lengths).std()

        plt.figure(figsize=(8,6))
        plt.bar(sample_nums.keys(), sample_nums.values(), color='skyblue')
        plt.xlabel('Dataset')
        plt.ylabel('Number of Samples')
        plt.title('Number of Samples per Dataset')
        plt.savefig(os.path.join(self.output_dir, 'sample_num.png'))
        plt.close()

        plt.figure(figsize=(8,6))
        plt.bar(q_length_means.keys(), q_length_means.values(), color='salmon')
        plt.xlabel('Dataset')
        plt.ylabel('Average Question Length')
        plt.title('Average Question Length per Dataset')
        plt.savefig(os.path.join(self.output_dir, 'q_length_mean.png'))
        plt.close()

        plt.figure(figsize=(8,6))
        plt.bar(q_length_stds.keys(), q_length_stds.values(), color='lightgreen')
        plt.xlabel('Dataset')
        plt.ylabel('Question Length Std Dev')
        plt.title('Question Length Standard Deviation per Dataset')
        plt.savefig(os.path.join(self.output_dir, 'q_length_std.png'))
        plt.close()

        print(f"Stat plots saved in '{self.output_dir}' directory.")

    def plot_start_word(self, top_n=5):
        start_word_data = {level: {ds: Counter() for ds in self.datasets} for level in range(1,4)}

        for ds in self.datasets:
            cqas = self._load_cqas(ds)
            for q in cqas:
                words = q[1].split()
                for level in range(1,4):
                    key = ' '.join(words[:level]) if len(words) >= level else ' '.join(words) + ' '*(level - len(words))
                    start_word_data[level][ds][key] += 1

        for level in range(1,4):
            plt.figure(figsize=(10,6))
            for ds in self.datasets:
                most_common = start_word_data[level][ds].most_common(top_n)
                labels, counts = zip(*most_common) if most_common else ([], [])
                plt.bar([f"{ds}_{label}" for label in labels], counts, alpha=0.7, label=ds)
            plt.xlabel('Starting Words')
            plt.ylabel('Frequency')
            plt.title(f'Top {top_n} Starting Words at Level {level}')
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'start_word_level_{level}.png'))
            plt.close()

        print(f"Start word plots saved in '{self.output_dir}' directory.")

    def plot_qtype(self):
        qtype_counts = {ds: Counter() for ds in self.datasets}

        for ds in self.datasets:
            qtype_path = shortcuts.get(ds, ds).replace("cqas", "qtype")
            if not os.path.exists(qtype_path):
                print(f"Qtype data for '{ds}' not found. Skipping.")
                continue
            with open(qtype_path, "rb") as f:
                qtypes = pickle.load(f)
            qtype_counts[ds] = Counter([qt[0] for qt in qtypes])

        df = pd.DataFrame(qtype_counts).fillna(0)
        df = df.astype(int)
        df = df.sort_index()
        df.plot(kind='bar', figsize=(12,8))
        plt.xlabel('Question Type')
        plt.ylabel('Count')
        plt.title('Question Type Distribution per Dataset')
        plt.legend(title='Dataset')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'qtype_distribution.png'))
        plt.close()

        print(f"Question type distribution plot saved in '{self.output_dir}' directory.")

    def plot_answerable(self, use_ctx=True, grouped=False):
        tag = "wc" if use_ctx else "woc"
        answerable_counts_stat = {}
        answerable_counts_non_stat = {}

        for ds in self.datasets:
            judge_ans_path = shortcuts.get(ds, ds).replace("cqas", f"gen_a_{tag}_star")
            if not os.path.exists(judge_ans_path):
                print(f"Judge answer data for '{ds}' not found. Skipping.")
                continue
            with open(judge_ans_path, "rb") as f:
                stars = pickle.load(f)
            cnt = Counter([x[0] if x[0] is not None else -1 for x in stars])
            answerable_counts_stat[ds] = cnt.get(1, 0)  # Assuming 1 represents answerable
            answerable_counts_non_stat[ds] = cnt.get(0, 0)  # Assuming 0 represents not answerable

        if grouped:
            labels = self.datasets
            stat_values = [answerable_counts_stat.get(ds, 0) for ds in labels]
            non_stat_values = [answerable_counts_non_stat.get(ds, 0) for ds in labels]

            x = range(len(labels))
            width = 0.35

            plt.figure(figsize=(10,6))
            plt.bar(x, stat_values, width, label='Stat Only', color='blue')
            plt.bar([p + width for p in x], non_stat_values, width, label='Non Stat Only', color='orange')

            plt.xlabel('Dataset')
            plt.ylabel('Count')
            plt.title('Answerable vs Non-Answerable Questions per Dataset')
            plt.xticks([p + width/2 for p in x], labels)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'answerable_grouped.png'))
            plt.close()
        else:
            plt.figure(figsize=(8,6))
            for ds in self.datasets:
                plt.bar(ds, answerable_counts_stat.get(ds, 0), alpha=0.7, label='Answerable' if ds == self.datasets[0] else "")
                plt.bar(ds, answerable_counts_non_stat.get(ds, 0), bottom=answerable_counts_stat.get(ds, 0), alpha=0.7, label='Non-Answerable' if ds == self.datasets[0] else "")
            plt.xlabel('Dataset')
            plt.ylabel('Count')
            plt.title('Answerable vs Non-Answerable Questions per Dataset')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'answerable.png'))
            plt.close()

        print(f"Answerable plots saved in '{self.output_dir}' directory.")

    def plot_len_req(self, grouped=False):
        min_len_stats = {}
        reduction_rates = {}

        for ds in self.datasets:
            min_ans_path = shortcuts.get(ds, ds).replace("cqas", "min_ans_len")
            if not os.path.exists(min_ans_path):
                print(f"Min answer length data for '{ds}' not found. Skipping.")
                continue
            with open(min_ans_path, "rb") as f:
                shorter = pickle.load(f)
            min_lengths = [x[0] for x in shorter]
            reduction = [x[0] / 10 for x in shorter]  # Example reduction rate
            min_len_stats[ds] = pd.Series(min_lengths).mean()
            reduction_rates[ds] = pd.Series(reduction).mean()

        if grouped:
            labels = self.datasets
            min_len_means = [min_len_stats.get(ds, 0) for ds in labels]
            reduction_means = [reduction_rates.get(ds, 0) for ds in labels]

            x = range(len(labels))
            width = 0.35

            plt.figure(figsize=(10,6))
            plt.bar(x, min_len_means, width, label='Min Answer Length', color='green')
            plt.bar([p + width for p in x], reduction_means, width, label='Reduction Rate', color='purple')

            plt.xlabel('Dataset')
            plt.ylabel('Values')
            plt.title('Min Answer Length and Reduction Rate per Dataset')
            plt.xticks([p + width/2 for p in x], labels)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'len_req_grouped.png'))
            plt.close()
        else:
            plt.figure(figsize=(8,6))
            plt.bar(min_len_stats.keys(), min_len_stats.values(), color='teal', label='Min Answer Length')
            plt.xlabel('Dataset')
            plt.ylabel('Average Min Answer Length')
            plt.title('Average Minimum Answer Length per Dataset')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'len_req.png'))
            plt.close()

        print(f"Length requirement plots saved in '{self.output_dir}' directory.")

    def plot_cover(self):
        coverage_data = {ds: {'word_total': [], 'sent_level': [], 'sent_cnt': []} for ds in self.datasets}

        for ds in self.datasets:
            cov_path = shortcuts.get(ds, ds).replace("cqas", "cov")
            if not os.path.exists(cov_path):
                print(f"Coverage data for '{ds}' not found. Skipping.")
                continue
            with open(cov_path, "rb") as f:
                cov = pickle.load(f)
            for entry in cov:
                coverage_data[ds]['word_total'].append(entry[0]['total'])
                coverage_data[ds]['sent_level'].append(entry[1])
                coverage_data[ds]['sent_cnt'].append(entry[2]['sents'])

        plt.figure(figsize=(10,6))
        for ds in self.datasets:
            if coverage_data[ds]['word_total']:
                plt.hist(coverage_data[ds]['word_total'], bins=30, alpha=0.5, label=ds)
        plt.xlabel('Total Word Coverage')
        plt.ylabel('Frequency')
        plt.title('Word Level Coverage Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'coverage_word_level.png'))
        plt.close()

        plt.figure(figsize=(10,6))
        for ds in self.datasets:
            if coverage_data[ds]['sent_cnt']:
                plt.hist(coverage_data[ds]['sent_cnt'], bins=30, alpha=0.5, label=ds)
        plt.xlabel('Sentence Count Coverage')
        plt.ylabel('Frequency')
        plt.title('Sentence Count Coverage Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'coverage_sent_cnt.png'))
        plt.close()

        print(f"Coverage plots saved in '{self.output_dir}' directory.")

    def plot_all(self, grouped_len_req=False, grouped_answerable=False):
        self.plot_stat()
        self.plot_start_word()
        self.plot_qtype()
        self.plot_answerable(grouped=grouped_answerable)
        self.plot_len_req(grouped=grouped_len_req)
        self.plot_cover()
        print("All plots have been generated and saved.")

if __name__ == "__main__":
    fire.Fire(PlotUtils)