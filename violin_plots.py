"""
22 Aug 2024
distribution of distances across source languages
python3 violin_plots.py --data uniq

"""

import sys
import pandas as pd
import argparse
import os
import time
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt

from configs import SLANG_MAP, MODEL, LANGUAGE_ORDER
import warnings

warnings.filterwarnings("ignore")


def make_dirs(logsto=None, picsto=None):
    os.makedirs(picsto, exist_ok=True)

    os.makedirs(logsto, exist_ok=True)
    current_datetime = datetime.utcnow()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H:%M')
    script_name = sys.argv[0].split("/")[-1].split(".")[0]
    log_file = f'{logsto}{formatted_datetime.split("_")[0]}_{script_name}.log'
    sys.stdout = Logger(logfile=log_file)
    print(f"\nRun date, UTC: {datetime.utcnow()}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")


class Logger(object):
    def __init__(self, logfile=None):
        self.terminal = sys.stdout
        self.log = open(logfile, "w")  # overwrite, don't "a" append

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', help="", required=False, default='data/final_wide_tables/')
    parser.add_argument('--data', choices=['uniq'], default='uniq')
    parser.add_argument('--model', choices=['rugpt3large_based_on_gpt2', 'ruRoBERTa-large', 'xlm-roberta-base'],
                        default=MODEL)

    parser.add_argument('--pics', default='pics/')
    parser.add_argument('--logs', default='logs/')
    parser.add_argument('--verbose', type=int, default=1)

    args = parser.parse_args()
    start = time.time()

    make_dirs(logsto=args.logs, picsto=args.pics)
    df = pd.read_csv(f"{args.indir}{args.data}/scores_ratios_responses_{args.model}.tsv", sep='\t')
    score_col = 'intelligibility'

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8))
    # Set the default font properties
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 16

    cmap = plt.get_cmap('RdYlGn')

    base_palette = [cmap(i) for i in range(0, 256, 256 // 5)]
    my_palette = {
        'Czech': base_palette[0],
        'Polish': base_palette[1],
        'Bulgarian': base_palette[2],
        'Belarusian': base_palette[3],
        'Ukrainian': base_palette[4]
    }
    df['language'] = df['language'].replace(SLANG_MAP)
    df['language'] = pd.Categorical(df['language'], categories=LANGUAGE_ORDER, ordered=True)

    # Create the violin plot
    sns.violinplot(x='language', y=score_col, data=df, inner='box', order=LANGUAGE_ORDER, palette=my_palette)

    # Add titles and labels
    plt.title('')
    plt.xlabel('')
    plt.ylabel(f'{score_col} scores', fontsize=17)
    plt.tick_params(axis='both', which='major', labelsize=17)  # major ticks

    # Show the plot
    plt.savefig('pics/scores_violins.png')
    if args.verbose:
        plt.show()
    else:
        plt.close()

    # look at the distribution of scores
    plt.figure(figsize=(10, 6))
    for language in LANGUAGE_ORDER:
        subset = df[df['language'] == language]
        sns.kdeplot(data=subset, x=score_col, label=language, fill=False, linewidth=2, color=my_palette[language])

    # Customize plot
    plt.legend(title="")
    plt.xlabel(score_col)
    plt.ylabel("Density")
    plt.title("")
    plt.savefig('pics/kde_scores.png')
    if args.verbose:
        plt.show()
    else:
        plt.close()
