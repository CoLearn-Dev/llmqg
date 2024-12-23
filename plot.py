import os
import fire
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
from collections import defaultdict
from run import CQA_Inspector

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

class PlotUtils:
    def __init__(self):
        self.inspector = CQA_Inspector()
        self.datasets = ["trivia", "hotpot", "llmqg_gpt", "llmqg_llama"]
    
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
        answerable_data_with = defaultdict(dict)
        answerable_data_without = defaultdict(dict)
        
        datasets = {
            "hotpot": "HotpotQA",
            "llmqg_llama": "Llama",
            "llmqg_gpt": "GPT-4o",
        }
        
        for dataset_key, dataset_label in datasets.items():
            results_with = self.inspector.answerable(dataset_key, use_ctx=True)
            for label, metrics in results_with.items():
                answerable_data_with[dataset_label][label] = metrics['percentage']
            
            results_without = self.inspector.answerable(dataset_key, use_ctx=False)
            for label, metrics in results_without.items():
                answerable_data_without[dataset_label][label] = metrics['percentage']
        
        df_with = pd.DataFrame(answerable_data_with).transpose()
        df_without = pd.DataFrame(answerable_data_without).transpose()

        df_with = df_with[df_with.columns[::-1]]
        df_without = df_without[df_without.columns[::-1]]
        
        color_palette = ['#003262', '#3B7EA1', '#00B0DA', '#CFDD45', '#FDB515', '#C4820E']
        plt.rcParams.update({
            'font.size': 45,
        })
        
        fig, axes = plt.subplots(1, 2, figsize=(24, 10), sharey=True)

        # With Context
        df_with.plot(
            kind='bar',
            stacked=True,
            ax=axes[0],
            color=color_palette[:df_with.shape[1]],
            edgecolor='white'
        )
        axes[0].set_title('With Context')
        legend = axes[0].legend(title='Ratings', bbox_to_anchor=(1.05, 1), loc='upper left', reverse=True)
        legend.remove()
        axes[0].tick_params(axis='x', rotation=0)
        axes[0].set_ylim(0, 1.0)
        
        # Without Context
        df_without.plot(
            kind='bar',
            stacked=True,
            ax=axes[1],
            color=color_palette[:df_without.shape[1]],
            edgecolor='white'
        )
        axes[1].set_title('Without Context')
        axes[1].legend(title='Ratings', bbox_to_anchor=(1.05, 1), loc='upper left', reverse=True)
        axes[1].tick_params(axis='x', rotation=0)
        axes[1].set_ylim(0, 1.0)
        
        plt.tight_layout()
        plot_path = os.path.join(PLOT_DIR, "answerable_distribution.png")
        plt.savefig(plot_path, bbox_inches='tight')
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
        datasets = ["hotpot", "llmqg_llama", "llmqg_gpt"]
        dataset_labels = {
            "hotpot": "HotpotQA",
            "llmqg_llama": "Llama",
            "llmqg_gpt": "GPT-4o",
        }
        for dataset in datasets:
            results = self.inspector.cover(dataset)
            cover_data[dataset] = {
                "buckets": results['buckets'],
                "bucket_freq": results['bucket_freq'],
            }

        example_buckets = cover_data[datasets[0]]['buckets']
        bucket_labels = [f"{ll:.1f}-{rr:.1f}" for ll, rr in example_buckets]
        fig, axs = plt.subplots(len(datasets), 1, figsize=(10, 10), sharex=True)

        plt.rcParams.update({
            'font.size': 20,
        })

        if len(datasets) == 1:
            axs = [axs]

        for ax, dataset in zip(axs, datasets):
            bucket_freq = cover_data[dataset]['bucket_freq']
            ax.bar(bucket_labels, bucket_freq, color='skyblue', width=1.0)
            ax.set_title(dataset_labels[dataset])
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.yaxis.set_tick_params(labelsize=17)
            ax.set_ylim(0, 0.55)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.setp(axs[-1].get_xticklabels(), rotation=45, ha='center', fontsize=17)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "coverage_histogram.png"))
        plt.close()


    def plot_all(self):
        self.plot_stat()
        self.plot_start_word()
        self.plot_qtype()
        self.plot_answerable()
        self.plot_len_req(grouped=True)
        self.plot_cover()

if __name__ == "__main__":
    fire.Fire(PlotUtils)
