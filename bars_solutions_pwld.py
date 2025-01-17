"""
26 Sept 2024
we have a new literal and newly calculated distances (PWLD, WAS and LD)
re-using the previous script to plot updated data

26 Jul 2024
plot SL-grouped bars for colour-coded translation solutions -- I don't like stacked bars
['paraphrase', 'correct', 'fluent_literal', 'awkward_literal', 'fantasy', 'noise', 'empty']  by decreasing success+effort

columns language normed_annotation  absolute_frequency  percentage

python3 bars_solutions_pwld.py --overlay_distance

"""

import sys
import pandas as pd
import argparse
import os
import time
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from configs import SLANG_MAP, LANGUAGE_ORDER, ANNO_ORDER


def plot_anno_with_pwld(df_bars=None, df_lines=None, outf=None, picsto=None, my_metric=None,
                        anno_order=None, my_colours=None, show=None):
    # Create the bar plot with overlaid lineplot
    plt.subplots(figsize=(10, 8))
    sns.set_style("whitegrid")
    sns.set_context('paper')
    # Set the default font properties
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 16

    ax = sns.barplot(data=df_bars, x='language', y='percentage', hue='normed_annotation',
                     hue_order=anno_order, palette=my_colours)
    ax.set_xlabel('')
    ax.set_ylabel('Percentage')
    ax.set_title('')
    ax.tick_params(axis='both', labelsize=16)

    ax2 = ax.twinx()
    y_gold = y_translit = y_lit = y_tgt_literacy = y_tgt_translit = None

    if my_metric == 'pwld':
        y_gold = 'pwld_stimulus_gold'
        y_tgt_literacy = 'pwld_gold_lit'

    sns.lineplot(data=df_lines, x='language', y=y_gold, ax=ax2, color='black',
                 marker='o', errorbar=None, label=None, linewidth=2)

    sns.lineplot(data=df_lines, x='language', y=y_tgt_literacy, ax=ax2, color='blue',
                 marker='v', linestyle='--', errorbar=None, label=None, linewidth=2)

    ax2.set_ylabel('')

    bar_handles, bar_labels = ax.get_legend_handles_labels()  # Handles and labels for barplot
    # Manually create a handle for the lineplot
    line_handle1 = Line2D([0], [0], color='black', marker='o', label=y_gold)

    line_handle5 = Line2D([0], [0], color='blue', marker='v', linestyle='--', label=y_tgt_literacy)

    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.set_title('')
    ax2.tick_params(axis='y', labelsize=16)
    ax2.grid(False)

    # Combine the barplot and lineplot legend items
    legend = ax.legend(handles=bar_handles + [line_handle1, line_handle5],  # line_handle2, line_handle3, line_handle4,
                       labels=bar_labels + [y_gold, y_tgt_literacy],  # y_translit, y_tgt_translit,
                       loc='upper right',  # , bbox_to_anchor=(1.4, 1) # Position the legend outside the plot
                       fontsize=12,
                       ncol=2,  # Set to 3 columns
                       frameon=False  # Remove the frame around the legend
                       )
    legend.get_frame().set_alpha(0.5)

    plt.tight_layout()

    plt.savefig(f'{picsto}{outf}')

    if show:
        plt.show()


def plot_anno(_df=None, outf=None, picsto=None, anno_order=None, my_colours=None, show=None):
    # Create the bar plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.set_context('paper')
    # Set the default font properties
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 14

    ax = sns.barplot(data=_df, x='language', y='percentage', hue='normed_annotation', hue_order=anno_order,
                     palette=my_colours)
    # Add labels and title
    plt.xlabel('')
    plt.ylabel('Percentage')
    plt.title('')
    plt.legend(title='', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

    plt.xticks(fontsize=14)  # Adjust the font size of x-axis tick labels
    plt.yticks(fontsize=14)

    # Show the plot
    plt.tight_layout()

    plt.savefig(f'{picsto}{outf}')

    datato = f'{picsto}tables/'
    os.makedirs(datato, exist_ok=True)
    df.to_csv(f'{datato}plot_normed_anno.tsv', sep='\t', index=False)
    if show:
        plt.show()


def loop_langs_and_serialise_dist(matrix_df, uniq_df):
    collector = []
    dist_columns = [d for d in uniq_df.columns.tolist() if d.startswith('pwld')]
    # print(dist_columns)
    for lang in matrix_df.language.unique():
        this_matrix_df = matrix_df[matrix_df['language'] == lang]

        this_filler_data = uniq_df[uniq_df['language'] == lang]

        for dist in dist_columns:
            this_dist_map = this_filler_data.set_index('phrase')[dist].to_dict()
            # print(*(f"{k}: {v}" for k, v in list(this_dist_map.items())[:2]), sep='\n')

            # this_matrix_df = this_matrix_df.assign(**{dist: this_matrix_df['src'].map(this_dist_map)})
            _this_matrix_df = this_matrix_df.copy()
            _this_matrix_df.loc[:, dist] = this_matrix_df['phrase'].map(this_dist_map)

            collector.append(_this_matrix_df)

    matrix_redone = pd.concat(collector, axis=0)

    return matrix_redone


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
    parser = argparse.ArgumentParser(description='barplots with overlay distances')
    parser.add_argument('--data', help="", default='data/normalised_translation_experiment_data.tsv')
    parser.add_argument('--anno_stats', help="", default='res/anno/overview_annotation_stats.tsv')
    parser.add_argument('--distances', help="", default='data/input/data_with_gptlit_translit_distances.csv')
    parser.add_argument('--overlay_distance', help="", action='store_true')
    parser.add_argument('--pics', default="pics/")
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()
    start = time.time()

    make_dirs(logsto=args.logs, picsto=args.pics)

    df = pd.read_csv(args.anno_stats, sep='\t')
    print(df.columns.tolist())
    print(df.head())
    print(df.shape)

    cmap = plt.get_cmap('RdYlGn')
    # Generate 7 distinct colors from the colormap
    base_palette = [cmap(i) for i in range(0, 256, 256 // 7)][::-1]
    custom_palette = {
        'correct': base_palette[0],
        'fluent_literal': base_palette[1],
        'paraphrase': base_palette[2],
        'awkward_literal': base_palette[3],
        'fantasy': base_palette[4],
        'noise': base_palette[5],
        'empty': base_palette[6]
    }

    outf = 'overview_annotation.png'
    df['language'] = df['language'].replace(SLANG_MAP)
    df['language'] = pd.Categorical(df['language'], categories=LANGUAGE_ORDER, ordered=True)

    plot_anno(_df=df, outf=outf, picsto=args.pics, anno_order=ANNO_ORDER, my_colours=custom_palette, show=args.verbose)

    if args.overlay_distance:
        existing_metrics = []
        dist_from = pd.read_csv(args.distances, sep=',')
        anno_from = pd.read_csv(args.data, sep='\t')  #

        dist_df = loop_langs_and_serialise_dist(anno_from, dist_from)
        # print(dist_df.head())

        # slang_map = {'CS': 'Czech', 'PL': 'Polish', 'BG': 'Bulgarian', 'UK': 'Ukrainian', 'BE': 'Belarusian'}
        dist_df['language'] = dist_df['language'].replace(SLANG_MAP)

        dist_df['language'] = pd.Categorical(dist_df['language'], categories=LANGUAGE_ORDER, ordered=True)

        # I cannot have  're_normed_annotation' on the x-axis taken by language!
        # _cols = ['language'] + [d for d in dist_df.columns.tolist() if d.startswith('pwld')]
        # numeric_cols_df = dist_df[_cols]

        score_avg = dist_df.groupby(['language']).mean(numeric_only=True).reset_index()
        dist_avg = score_avg.rename(columns={'pwld_gold_original': 'pwld_stimulus_gold'})
        outf = 'pwld+annotation.png'

        df['language'] = pd.Categorical(df['language'], categories=LANGUAGE_ORDER, ordered=True)

        for plottable in ['pwld']:  # 'pwld'
            plot_anno_with_pwld(df_bars=df, df_lines=dist_avg, outf=outf, picsto=args.pics, my_metric=plottable,
                                anno_order=ANNO_ORDER, my_colours=custom_palette)

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
