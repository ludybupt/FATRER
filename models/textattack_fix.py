from cmath import exp
from textattack.loggers.attack_log_manager import *
from textattack.loggers.weights_and_biases_logger import *
from textattack.shared.utils import LazyLoader, html_table_from_rows

'''
import wandb
def wandb_log_summary_rows(self, rows, title, window_id):
    table = wandb.Table(columns=["Attack Results", "score"])
    for row in rows:
        if isinstance(row[1], str):
            try:
                row[1] = row[1].replace("%", "")
                row[1] = float(row[1])
            except ValueError:
                raise ValueError(
                    f'Unable to convert row value "{row[1]}" for Attack Result "{row[0]}" into float'
                )
        print('row', row)
        print('*row', *row)
        table.add_data(*row)
        metric_name, metric_score = row
        wandb.run.summary[metric_name] = metric_score
    wandb.log({"attack_params": table})
'''
def get_log_summary(self):
    total_attacks = len(self.results)
    if total_attacks == 0:
        return

    # Default metrics - calculated on every attack
    attack_success_stats = AttackSuccessRate().calculate(self.results)
    words_perturbed_stats = WordsPerturbed().calculate(self.results)
    attack_query_stats = AttackQueries().calculate(self.results)

    # @TODO generate this table based on user input - each column in specific class
    # Example to demonstrate:
    # summary_table_rows = attack_success_stats.display_row() + words_perturbed_stats.display_row() + ...
    summary_table_rows = [
        [
            "Number of successful attacks:",
            attack_success_stats["successful_attacks"],
        ],
        ["Number of failed attacks:", attack_success_stats["failed_attacks"]],
        ["Number of skipped attacks:", attack_success_stats["skipped_attacks"]],
        [
            "Original accuracy:",
            str(attack_success_stats["original_accuracy"]) + "%",
        ],
        [
            "Accuracy under attack:",
            str(attack_success_stats["attack_accuracy_perc"]) + "%",
        ],
        [
            "Attack success rate:",
            str(attack_success_stats["attack_success_rate"]) + "%",
        ],
        [
            "Average perturbed word %:",
            str(words_perturbed_stats["avg_word_perturbed_perc"]) + "%",
        ],
        [
            "Average num. words per input:",
            words_perturbed_stats["avg_word_perturbed"],
        ],
    ]

    summary_table_rows.append(
        ["Avg num queries:", attack_query_stats["avg_num_queries"]]
    )

    if self.enable_advance_metrics:
        perplexity_stats = Perplexity().calculate(self.results)
        use_stats = USEMetric().calculate(self.results)

        summary_table_rows.append(
            [
                "Average Original Perplexity:",
                perplexity_stats["avg_original_perplexity"],
            ]
        )

        summary_table_rows.append(
            [
                "Average Attack Perplexity:",
                perplexity_stats["avg_attack_perplexity"],
            ]
        )
        summary_table_rows.append(
            ["Average Attack USE Score:", use_stats["avg_attack_use_score"]]
        )

    self.log_summary_rows(
        summary_table_rows, "Attack Results", "attack_results_summary"
    )
    # Show histogram of words changed.
    numbins = max(words_perturbed_stats["max_words_changed"], 10)
    for logger in self.loggers:
        logger.log_hist(
            words_perturbed_stats["num_words_changed_until_success"][:numbins],
            numbins=numbins,
            title="Num Words Perturbed",
            window_id="num_words_perturbed",
        )
    process_result = []
    for items in summary_table_rows:
        title = items[0].replace(':','')
        score = 0
        if isinstance(items[1], str):
            if '%' in items[1]:
                score_str = items[1].replace('%','')
                try:
                    score = float(score_str)/100
                except:
                    score = 0
                    print('nan error', score_str)
        else:
            score = float(items[1])
        process_result.append([title, score]) 
        
    return summary_table_rows, process_result

