"""
16 Jan 2025 UPD: Figure 2.
17 Oct 2025
take a sample of items in each language where gold=literal or top third based on PWLD between orig=gold

plot the distribution of translation solutions for this sample with a superimposed lineplot for intelligibility scores averaged across these samples

python3 literality_bars.py --overlay --strategy pwld_top33
python3 literality_bars.py --overlay --strategy gold=lit

"""

import sys
from collections import defaultdict

import pandas as pd
import argparse
import os
import time
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from configs import SLANG_MAP, MODEL, ANNO_ORDER


def plot_anno_with_pwld(df_bars=None, df_lines=None, outf=None, picsto=None, my_metric=None,
                        anno_order=None, my_colours=None, show=None):
    # Create the bar plot with overlaid lineplot
    plt.subplots(figsize=(10, 10))
    sns.set_style("whitegrid")
    sns.set_context('paper')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 16

    ax = sns.barplot(data=df_bars, x='language', y='percentage', hue='normed_annotation',
                     hue_order=anno_order, palette=my_colours, zorder=2)
    ax.set_xlabel('')
    ax.set_ylabel('Percentage', fontdict={'fontsize': 16})
    ax.set_title('')
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, axis='y', zorder=0)
    ax2 = ax.twinx()

    sky_blue = '#87CEEB'

    sns.lineplot(data=df_lines, x='language', y=my_metric, ax=ax2, color=sky_blue,
                 marker='o', errorbar=None, label=None, linewidth=2, zorder=3)
    sns.lineplot(data=df_lines, x='language', y='pwld_stimulus_gold', ax=ax2, color='black',
                 marker='v', errorbar=None, label=None, linewidth=2, zorder=3)

    ax2.set_ylabel('PWLD and intelligibility scores', fontdict={'fontsize': 16})

    bar_handles, bar_labels = ax.get_legend_handles_labels()  # Handles and labels for barplot
    # Manually create a handle for the lineplot
    line_handle1 = Line2D([0], [0], color=sky_blue, marker='o', label=my_metric)
    line_handle2 = Line2D([0], [0], color='black', marker='v', label='pwld_stimulus_gold')

    ax2.set_xlabel('')
    ax2.set_title('')
    ax2.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax2.grid(False)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(
        handles=bar_handles + [line_handle1, line_handle2],  # Add other line handles here if needed
        labels=bar_labels + [my_metric, 'pwld_stimulus_gold'],  # Add other labels here if needed
        loc='upper center',  # Center the legend horizontally
        bbox_to_anchor=(0.5, -0.2),  # Position the legend below the plot
        ncol=5,  # Set to 3 columns
        fontsize=12,
        frameon=False  # Remove the frame around the legend
    )
    # Show the plot
    plt.tight_layout()

    plt.savefig(f'{picsto}{outf}')
    if show:
        plt.show()
    else:
        plt.close()


def plot_anno(_df=None, outf=None, picsto=None, anno_order=None, my_colours=None, show=None):
    # Create the bar plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.set_context('talk')
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
    else:
        plt.close()


def loop_langs_and_serialise_dist(matrix_df, uniq_df, my_col=None):
    collector = []
    for lang in matrix_df.language.unique():
        this_df = matrix_df[matrix_df['language'] == lang]
        this_filler_data = uniq_df[uniq_df['language'] == lang]
        this_dist_map = this_filler_data.set_index('src')[my_col].to_dict()

        # print(*(f"{k}: {v}" for k, v in list(this_dist_map.items())[:2]), sep='\n')

        # this_matrix_df = this_matrix_df.assign(**{dist: this_matrix_df['src'].map(this_dist_map)})
        _this_df = this_df.copy()
        _this_df.loc[:, my_col] = this_df['src'].map(this_dist_map)

        collector.append(_this_df)

    matrix_redone = pd.concat(collector, axis=0)

    return matrix_redone


def get_anno_stats(data=None, anno_col=None, lang=None):
    # Count values in normed_annotation column
    annotation_counts = data[anno_col].value_counts().reset_index()
    annotation_counts.columns = [anno_col, 'absolute_frequency']

    # print(annotation_counts)
    # exit()

    # Calculate the percentage
    total = len(data)
    annotation_counts['percentage'] = (annotation_counts['absolute_frequency'] / total) * 100
    # Format the percentage to 2 decimal places
    annotation_counts['percentage'] = annotation_counts['percentage'].apply(lambda x: float(f"{x:.2f}"))

    annotation_counts.insert(0, 'language', lang)

    # Create a DataFrame for the new row
    tot_row = pd.DataFrame({
        'language': [lang],
        anno_col: ['Total'],
        'absolute_frequency': [total],
        'percentage': [100]
    })

    # Concatenate the new row with the existing DataFrame
    anno_df = pd.concat([annotation_counts, tot_row], ignore_index=True)

    return anno_df


def get_count(my_df=None, col_name=None):
    if col_name == 'tot':
        langs = defaultdict()
        for l in my_df.language.unique():
            my_df_l = my_df[my_df.language == l]
            langs[l] = my_df_l.shape[0]
        df_count = pd.DataFrame(langs, index=[0]).T.reset_index()
    else:
        # Step 1: Extract the portion of the ID before the colon
        my_df[col_name] = my_df['ID'].apply(lambda x: x.split(':')[0])

        # Step 2: Group by language and count unique ID_prefix values
        df_count = my_df.groupby('language')[col_name].nunique().reset_index()

    # Step 3: Rename the count column
    df_count.columns = ['language', col_name]
    df_count = df_count.set_index('language')

    return df_count


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='is literary translation a good strategy?')
    parser.add_argument('--intab', help="", default=f'data/final_wide_tables/user/features21_{MODEL}.tsv')
    parser.add_argument('--user_strings', help="", default='data/final_wide_tables/user/user_string_columns.tsv')
    parser.add_argument('--overlay', help="", action='store_true')
    parser.add_argument('--overlay_data', help="", default=f'data/final_wide_tables/uniq/scores_ratios_responses_{MODEL}.tsv')
    parser.add_argument('--strategy', choices=['gold=lit', 'pwld_top50', 'pwld_top33'], default="pwld_top33")
    parser.add_argument('--pics', default="pics/literal_strategy/")
    parser.add_argument('--logs', default='logs/')
    parser.add_argument('--verbose', type=int, default=1)

    args = parser.parse_args()

    start = time.time()

    make_dirs(logsto=args.logs, picsto=args.pics)

    df = pd.read_csv(args.intab, sep='\t').set_index('ID')

    str_df = pd.read_csv(args.user_strings, sep='\t').set_index('ID')
    str_df = str_df.drop(['language', 'category', 'normed_annotation', 'src', 'src_sent',
                          'gold_sent', 'lit_sent', 'user_sent'], axis=1)
    _df = pd.concat([df, str_df], axis=1)
    _df = _df.reset_index()

    min_value = _df['pwld_gold_original'].min()
    max_value = _df['pwld_gold_original'].max()

    print("Minimum value pwld_gold_original:", min_value)
    print("Maximum value pwld_gold_original:", max_value)
    # Minimum value pwld_gold_original: 0.0263157894736842: UK  не раз  не раз  не раз; BE  можна сказаць  можно сказать  можно сказать 0.035885
    # CS       na svete       на свете         в мире            0.045113
    # Maximum value pwld_gold_original: 0.8789473684210526: PL  niejednokrotnie  не раз  неоднократно
    # Get rows with the minimum and maximum values
    min_rows = _df[_df['pwld_gold_original'] < min_value][['language', 'src', 'gold', 'lit', 'pwld_gold_original']]
    max_rows = _df[_df['pwld_gold_original'] > max_value][['language', 'src', 'gold', 'lit', 'pwld_gold_original']]

    u_min_rows = min_rows.drop_duplicates(subset='src', keep='first')
    u_max_rows = max_rows.drop_duplicates(subset='src', keep='first')

    print("Rows with 'pwld_gold_original' close to minimum value:")  # most similar items
    print(u_min_rows)

    print("\nRows with 'pwld_gold_original' close to maximum value:")  # most distant items
    print(u_max_rows)

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

    slang_anno_stats_collector = []
    all_literal_samples = []
    for slang in _df['language'].unique():
        print(slang)
        upd_this_lang = _df[_df.language == slang]

        if args.strategy == 'gold=lit':
            literal_sample = upd_this_lang[upd_this_lang['gold'] == upd_this_lang['lit']]
        else:
            pwld_sorted = upd_this_lang.sort_values(by='pwld_gold_original', ascending=True)

            if args.strategy.endswith('50'):
                # Calculate the 66th percentile to get the threshold for the top third
                threshold = pwld_sorted['pwld_gold_original'].quantile(0.50)
                min_value = pwld_sorted['pwld_gold_original'].min()
                max_value = pwld_sorted['pwld_gold_original'].max()

                print("Minimum value pwld_gold_original:", min_value)
                print("Maximum value pwld_gold_original:", max_value)
                print(f"Threshold for {50}% of lowest pwld_gold_original values for {slang}: {threshold}")
            else:
                threshold = pwld_sorted['pwld_gold_original'].quantile(0.33)
                min_value = pwld_sorted['pwld_gold_original'].min()
                max_value = pwld_sorted['pwld_gold_original'].max()

                print("Minimum value pwld_gold_original:", min_value)
                print("Maximum value pwld_gold_original:", max_value)
                print(f"Threshold for {33}% of the lowest pwld_gold_original values for {slang}: {threshold}")

                similar_rows = pwld_sorted[pwld_sorted['pwld_gold_original'] < threshold][
                    ['language', 'src', 'gold', 'lit', 'pwld_gold_original']]
                u_similar_rows = similar_rows.drop_duplicates(subset='src', keep='first')
                print(f"Rows with 'pwld_gold_original' below {threshold}:")  ## most similar items
                print(u_similar_rows.head())

            literal_sample = pwld_sorted[pwld_sorted['pwld_gold_original'] < threshold]

        this_anno_stats_df = get_anno_stats(data=literal_sample, anno_col='normed_annotation', lang=slang)

        all_literal_samples.append(literal_sample)
        slang_anno_stats_collector.append(this_anno_stats_df)

    anno2plot = pd.concat(slang_anno_stats_collector, axis=0)

    literal_df = pd.concat(all_literal_samples, axis=0)

    if args.overlay:
        existing_metrics = []
        intel_df = pd.read_csv(args.overlay_data, sep='\t')

        my_score_mean_lang_df = loop_langs_and_serialise_dist(literal_df, intel_df, my_col='intelligibility')
        my_score_mean_lang_df['language'] = my_score_mean_lang_df['language'].replace(SLANG_MAP)

        intel_df['language'] = intel_df['language'].replace(SLANG_MAP)
        tot_df_count = get_count(my_df=intel_df, col_name='tot')
        lit_df_count = get_count(my_df=my_score_mean_lang_df, col_name='gold=lit')

        lang_string = pd.concat([lit_df_count, tot_df_count], axis=1)
        str_lst = lang_string.apply(lambda x: f"({x['gold=lit']}/{x['tot']})", axis=1)
        lang_string.insert(0, 'lang_string', str_lst)

        # this is where the intelligibility is averaged across items with literal potential
        _avg = my_score_mean_lang_df.groupby(['language']).mean(numeric_only=True)

        updated_avg = pd.concat([_avg, lang_string['lang_string']], axis=1).reset_index()
        key_val_dict = dict(zip(updated_avg['language'], updated_avg['lang_string']))
        new_avg = updated_avg.copy()
        # new_avg['language'] = pd.Categorical(new_avg['language'], categories=LANGUAGE_ORDER, ordered=True)
        new_avg['language'] = updated_avg.apply(lambda x: '\n'.join([x['language'], x['lang_string']]), axis=1)

        # print(new_avg.head())
        # print(new_avg.columns.tolist())
        # print(args.strategy)

        outf = f'literal+intelligibility-{args.strategy}.png'
        anno2plot['language'] = anno2plot['language'].replace(SLANG_MAP)
        anno2plot['language'] = anno2plot.apply(lambda x: '\n'.join([x['language'], key_val_dict[x['language']]]), axis=1)
        print(anno2plot.head(15))
        print(set(anno2plot['language'].tolist()))

        # all responses coinciding with gold are coded as fluent\_literal,
        # so the correct solution does not exist for this sample
        if args.strategy == 'gold=lit':
            ANNO_ORDER = [x for x in ANNO_ORDER if x != 'correct']

        if args.strategy == 'gold=lit':
            NEW_LANGUAGE_ORDER = ['Czech\n(14/60)', 'Polish\n(16/50)', 'Bulgarian\n(27/56)', 'Belarusian\n(31/57)',
                                  'Ukrainian\n(36/59)']
        elif args.strategy.endswith('50'):
            NEW_LANGUAGE_ORDER = ['Czech\n(33/60)', 'Polish\n(24/50)', 'Bulgarian\n(27/56)', 'Belarusian\n(28/57)',
                                  'Ukrainian\n(31/59)']
        else:
            # # for top 1/3
            NEW_LANGUAGE_ORDER = ['Czech\n(21/60)', 'Polish\n(16/50)', 'Bulgarian\n(17/56)', 'Belarusian\n(18/57)',
                                  'Ukrainian\n(20/59)']
            # for the upper half of items sorted ascendingly (it is a distance!)

        new_avg['language'] = pd.Categorical(new_avg['language'], categories=NEW_LANGUAGE_ORDER, ordered=True)
        anno2plot['language'] = pd.Categorical(anno2plot['language'], categories=NEW_LANGUAGE_ORDER, ordered=True)
        new_avg = new_avg.rename(columns={'pwld_gold_original': 'pwld_stimulus_gold'})
        plot_anno_with_pwld(df_bars=anno2plot, df_lines=new_avg, outf=outf, picsto=args.pics, my_metric='intelligibility',
                            anno_order=ANNO_ORDER, my_colours=custom_palette, show=args.verbose)

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
