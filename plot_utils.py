import os
import fire
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from run import CQA_Inspector

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

class PlotUtils:
    def __init__(self):
        self.inspector = CQA_Inspector()
        self.datasets = ["trivia", "hotpot", "llmqg"]
    
    def plot_stat(self):
        stats_data = {}
        for dataset in self.datasets:
            stats = self.inspector.stat(dataset)
            stats_data[dataset] = stats
        stats_df = pd.DataFrame(stats_data).transpose()
        stats_to_plot = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
        stats_subset = stats_df[stats_to_plot]

        plt.rcParams.update({'font.size': 14})
        ax = stats_subset['mean'].plot(kind='bar', yerr=stats_subset['std'], figsize=(10, 7), capsize=4)
        plt.title('Question Length Statistics')
        plt.ylabel('Length')
        plt.xlabel('Dataset')
        plt.xticks(rotation=0)
        plt.legend(title='Statistics')
        plt.tight_layout()

        plot_path = os.path.join(PLOT_DIR, "question_length_stats.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to {plot_path}")

    def plot_start_word(self, top_n=10):
        for dataset in self.datasets:
            results = self.inspector.start_word(dataset)
            first_level = results.get(1, {})
            top_words = sorted(first_level.items(), key=lambda x: x[1], reverse=True)[:top_n]
            words, frequencies = zip(*top_words)

            plt.figure(figsize=(10, 6))
            sns.barplot(x=list(words), y=list(frequencies))
            plt.title(f'Top {top_n} Starting Words - {dataset.capitalize()}')
            plt.xlabel('Starting Word')
            plt.ylabel('Frequency (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()

            plot_path = os.path.join(PLOT_DIR, f"top_{top_n}_starting_words_{dataset}.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot to {plot_path}")

    def plot_qtype(self):
        qtype_data = {}
        for dataset in self.datasets:
            results = self.inspector.qtype(dataset)
            qtypes = list(results.keys())
            percentages = [results[qt]['percentage'] for qt in qtypes]
            qtype_data[dataset] = percentages

        qtype_df = pd.DataFrame(qtype_data, index=qtypes)

        qtype_df.plot(kind='bar', figsize=(12, 8))
        plt.title('Question Type Distribution')
        plt.xlabel('Question Type')
        plt.ylabel('Percentage')
        plt.xticks(rotation=45)
        plt.legend(title='Dataset')
        plt.tight_layout()

        plot_path = os.path.join(PLOT_DIR, "question_type_distribution.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to {plot_path}")

    def plot_answerable(self):
        answerable_data = defaultdict(dict)
        for dataset in self.datasets:
            results = self.inspector.answerable(dataset, use_ctx=True)
            for label, metrics in results.items():
                answerable_data[dataset][label] = metrics['percentage']

        answerable_df = pd.DataFrame(answerable_data).transpose()
        answerable_df.plot(kind='bar', stacked=True, figsize=(10, 7))
        plt.title('Answerable Distribution')
        plt.xlabel('Dataset')
        plt.ylabel('Percentage')
        plt.xticks(rotation=0)
        plt.legend(title='Answerable Tags', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        plot_path = os.path.join(PLOT_DIR, "answerable_distribution.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to {plot_path}")

    def plot_len_req(self, grouped=False):
        len_req_data = {}
        len_req_data_grouped = {}
        for dataset in self.datasets:
            stats = self.inspector.len_req(dataset, stat_only=False)
            print(f"Answer Length Stats - {dataset} - {stats}")
            len_req_data[dataset] = stats["minimize_answer_length_stats"]
            len_req_data_grouped[dataset] = stats["reduction_rate_stats"]

            if grouped:
                stats_stat_only = self.inspector.len_req(dataset, stat_only=True)
                if stats_stat_only:
                    len_req_data_grouped[f"{dataset}_stat_only"] = stats_stat_only["minimize_answer_length_stats"]

        means = {k: v.get('mean', 0) for k, v in len_req_data.items()}
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(means.keys()), y=list(means.values()))
        plt.title('Mean Answer Length')
        plt.xlabel('Dataset')
        plt.ylabel('Mean Words')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plot_path = os.path.join(PLOT_DIR, "mean_answer_length.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to {plot_path}")

        if grouped:
            reduction_rates = {}
            for k, v in len_req_data_grouped.items():
                print(f"Reduction Rate Stats - {k} - {v}")
                reduction_rate = v.get('mean', 0)
                reduction_rates[k] = reduction_rate

            plt.figure(figsize=(10, 6))
            sns.barplot(x=list(reduction_rates.keys()), y=list(reduction_rates.values()))
            plt.title('Mean Reduction Rate')
            plt.xlabel('Dataset')
            plt.ylabel('Reduction Rate')
            plt.xticks(rotation=45)
            plt.tight_layout()

            plot_path = os.path.join(PLOT_DIR, "mean_reduction_rate.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot to {plot_path}")

    def plot_cover(self):
        cover_data = {}
        for dataset in self.datasets:
            results = self.inspector.cover(dataset)
            cover_data[dataset] = {
                "word_level_mean": results['word_level_stats']['mean'],
                "word_cnt_mean": results['word_cnt_stats']['mean'],
                "sent_level_mean": results['sent_level_stats']['mean'],
                "sent_cnt_mean": results['sent_cnt_stats']['mean']
            }
        cover_df = pd.DataFrame(cover_data).transpose()
        fig, ax = plt.subplots(2, 1, figsize=(12, 12))

        cover_df[['word_level_mean', 'sent_level_mean']].plot(kind='bar', ax=ax[0])
        ax[0].set_title('Coverage Level Statistics')
        ax[0].set_xlabel('Dataset')
        ax[0].set_ylabel('Mean Coverage')
        ax[0].legend(['Word Level', 'Sentence Level'])

        cover_df[['word_cnt_mean', 'sent_cnt_mean']].plot(kind='bar', ax=ax[1])
        ax[1].set_title('Coverage Count Statistics')
        ax[1].set_xlabel('Dataset')
        ax[1].set_ylabel('Mean Count')
        ax[1].legend(['Word Count', 'Sentence Count'])

        plt.tight_layout()

        plot_path = os.path.join(PLOT_DIR, "coverage_statistics.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to {plot_path}")

    def plot_all(self):
        self.plot_stat()
        self.plot_start_word()
        self.plot_qtype()
        self.plot_answerable()
        self.plot_len_req(grouped=True)
        self.plot_cover()

if __name__ == "__main__":
    fire.Fire(PlotUtils)
