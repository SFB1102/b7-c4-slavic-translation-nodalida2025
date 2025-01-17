"""
16 Jan UPD:
Tables 4a and 4b in the appendix

python3 univariate_analysis.py
"""

import sys
from collections import defaultdict, Counter

import pandas as pd
import argparse
import os
import time
from datetime import datetime

from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from ast import literal_eval

from configs import SLANG_MAP, MODEL, LANGUAGE_ORDER, PREDICTORS
import warnings

warnings.filterwarnings("ignore")


def plot_collinearity(my_feats_df, lang=None, save_as=None, verbosity=None):
    # Calculate the correlation matrix
    correlation_matrix = my_feats_df.corr()
    print(correlation_matrix.shape)

    # Reorder the features based on correlation
    # (optional, you can remove this if you don't want to reorder)
    feature_order = correlation_matrix.mean().sort_values().index
    reordered_corr_matrix = correlation_matrix[feature_order].reindex(feature_order)

    # Create a heatmap
    plt.figure(figsize=(12, 9))

    ax = sns.heatmap(reordered_corr_matrix, annot=False, cmap='coolwarm', fmt='.2f', square=True, linewidth=.5)

    # # x-labels to the top axis
    # ax.set(xlabel="", ylabel="")
    # ax.xaxis.tick_top()

    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.title(f'{lang}')

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    plt.savefig(save_as)
    if verbosity:
        plt.show()
    plt.close()


def sorted_correlations(my_df=None, x_features=None, y=None, metric=None):
    res_dict = defaultdict(list)
    for predictor in x_features:
        pred_vals = my_df[predictor].values
        y_vals = my_df[y].values
        res, p_string = get_corr(pred=pred_vals, resp=y_vals, metric=metric)

        res_dict[predictor].append(res)
        res_dict[predictor].append(p_string)

    res_df = pd.DataFrame.from_dict(res_dict, orient='index').reset_index()

    # Rename columns (optional, for better clarity)
    res_df.columns = ['feature', 'correlation', 'p-value']
    res_df = res_df.sort_values(by='correlation', key=abs, ascending=False)

    return res_df


def get_corr(pred=None, resp=None, metric=None):
    if metric == 'spearman':
        corr, p = spearmanr(pred, resp, nan_policy='omit')
    else:
        corr, p = pearsonr(pred, resp)

    return corr, f'{p:.4f}'


def plot_correlations_feats(my_df=None, corr_res=None, y=None, lang=None, outpath=None, verbosity=False):
    x_features = corr_res['feature'].tolist()
    corr_vals = corr_res['correlation']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(my_df[x_features])  # np.array
    scaled_y = scaler.fit_transform(my_df[[y]])
    sns.set(style="whitegrid")
    palette = sns.color_palette("husl", len(corr_res))

    plt.figure(figsize=(12, 9))
    handles = []
    labels = []
    y_vals = scaled_y.flatten()  # Flatten to get 1D array for plotting
    for feat, corr_val, col in zip(x_features, corr_vals, palette):
        pred_vals = scaled_features[:, x_features.index(feat)]
        # pred_vals = my_df[feat].values
        # y_vals = my_df[y].values
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 8))
        # Set the default font properties
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 16
        sns.regplot(x=pred_vals, y=y_vals, data=my_df, scatter=True, color=col,
                    line_kws={'linewidth': 2}, label=feat)

        # Append the handle (plot line) for the legend
        handles.append(plt.Line2D([0], [0], color=col, lw=2))
        labels.append(f'{feat} ({corr_val:.2f})')
    plt.legend(
        handles=handles, labels=labels,
        loc='upper right',
        title="",
        # bbox_to_anchor=(0.5, -0.2),  # Position the legend below the plot
        ncol=2,  # Set to 3 columns
        fontsize=16,
        frameon=False  # Remove the frame around the legend
    )
    plt.title(f"{lang}: Top {len(x_features)} features correlated with {y}")
    plt.ylabel(y, fontsize=17)
    plt.tick_params(axis='both', which='major', labelsize=17)  # major ticks
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig(outpath)

    if verbosity:
        plt.show()
    plt.close()


def plot_correlations_langs(my_df=None, scale_them=None, x=None, y=None, lang_map=None, outpath=None, show=None):
    scaler = StandardScaler()
    scaled_df = my_df[scale_them]
    scaled_df[scaled_df.columns] = scaler.fit_transform(scaled_df[scaled_df.columns])
    scaled_df.insert(0, 'language', my_df['language'].tolist())
    scaled_df['language'] = scaled_df['language'].replace(lang_map)
    language_order = ['Czech', 'Polish', 'Bulgarian', 'Belarusian', 'Ukrainian']
    # scaled_df['language'] = pd.Categorical(scaled_df['language'], categories=language_order, ordered=True)

    # sns.set(style="whitegrid")
    cmap = plt.get_cmap('RdYlGn')
    base_palette = [cmap(i) for i in range(0, 256, 256 // 5)]
    my_palette = {
        'Czech': base_palette[0],
        'Polish': base_palette[1],
        'Bulgarian': base_palette[2],
        'Belarusian': base_palette[3],
        'Ukrainian': base_palette[4]
    }
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set the default font properties
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 16
    # Plot each language in the specified order
    for lang in language_order:
        color = my_palette[lang]

        sns.regplot(x=x, y=y, data=scaled_df[scaled_df['language'] == lang], scatter=True, color=color,
                    line_kws={'linewidth': 2}, label=lang, ax=ax)

    # Plot linear regression for all data
    sns.regplot(x=x, y=y, data=scaled_df, scatter=False, color='black',
                line_kws={'linewidth': 2, 'alpha': 0.8}, label='all data', ax=ax)

    plt.legend(title='', loc='upper right', ncol=2,  # bbox_to_anchor=(1.05, 1.0),
               fontsize=17, title_fontsize=12, frameon=True, markerscale=1.5)
    sns.despine()
    # plt.grid(True, axis='y', zorder=1)
    plt.tight_layout()
    plt.title('')
    ax.set_ylabel(y, fontsize=17)
    plt.tick_params(axis='both', which='major', labelsize=17)  # major ticks
    # plt.ylabel(y, fontsize=17)
    plt.xlabel(x.replace('entropy', 'entropy of translation solutions'), fontsize=17)

    plt.savefig(outpath)
    if show:
        plt.show()


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
    parser.add_argument('--y_column', choices=['entropy', 'intelligibility'], default='entropy')
    parser.add_argument('--nbest', type=int, default=9, help="import feature names from the regression results")
    parser.add_argument('--N', type=int, default=7)
    parser.add_argument("--my_r", choices=['pearson', 'spearman'], help="which correlation coef to use",
                        default='pearson')
    parser.add_argument('--pics', default='pics/regplots/')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()
    start = time.time()
    make_dirs(logsto=args.logs, picsto=args.pics)

    meta = ['item_ID', 'language', 'src', 'category', args.y_column]

    _df = pd.read_csv(f"{args.indir}{args.data}/scores_ratios_responses_{args.model}.tsv", sep='\t')

    print(_df.columns.tolist())
    print(_df.shape)

    corr_collector = []

    for slang in _df.language.unique():
        full_slang = SLANG_MAP[slang]
        print(slang)
        df = _df[_df.language == slang]
        df = df.dropna()
        # avoid plotting everything! plot top most correlated feats for this lang
        correlates_df = sorted_correlations(my_df=df, x_features=PREDICTORS, y=args.y_column, metric=args.my_r)
        # correlates_df.insert(0, 'language', SLANG_MAP[slang])
        print(f"Total feats: {correlates_df.shape[0]}")

        print(correlates_df)
        correlates_df['correlation'] = correlates_df['correlation'].round(3)

        # report correlations for all results in Appendix
        # Convert 'p-value' column to numeric, coercing errors to NaN
        correlates_df['p-value'] = pd.to_numeric(correlates_df['p-value'], errors='coerce')

        # Drop rows where 'p-value' is NaN
        correlates_df = correlates_df.dropna(subset=['p-value'])

        correlates_df['correlation'] = correlates_df.apply(lambda row: f"{row['correlation']}*" if row['p-value'] < 0.05 else row['correlation'], axis=1)
        correlates_df = correlates_df.rename(columns={'correlation': SLANG_MAP[slang]}).set_index('feature').drop(['p-value'], axis=1)
        corr_collector.append(correlates_df)

    total_corr_res_df = pd.concat(corr_collector, axis=1)
    total_corr_res_df = total_corr_res_df[LANGUAGE_ORDER].reset_index()

    # Function to count columns without asterisks in a row
    def count_without_asterisks(row):
        return sum(1 for value in row if isinstance(value, (int, float)) or ('*' not in str(value)))

    # Add a column for counts without asterisks
    total_corr_res_df['no_asterisks'] = total_corr_res_df.apply(count_without_asterisks, axis=1)

    # Sort the DataFrame based on 'no_asterisks', descending
    df_sorted = total_corr_res_df.sort_values(by='no_asterisks', ascending=True).drop(columns=['no_asterisks'])

    df_sorted = df_sorted[df_sorted['feature'] != 'user_vars_num']
    df_sorted_reset = df_sorted.reset_index(drop=True)  # Reset the index
    df_sorted_reset.index = df_sorted_reset.index + 1  # Shift the index to start from 1
    df_sorted_reset.to_csv(f'res/total_correlations_{args.y_column}.tsv', sep='\t')
    print(df_sorted_reset)

    strong_x = ['entropy', 'pwld_original_gold', 'user_vars_num']
    _df = _df.rename(columns={'pwld_gold_original': 'pwld_original_gold'})
    # across languages
    if args.y_column == 'intelligibility':
        for feat in strong_x:
            if feat != 'entropy':
                continue

            langs_view = f'{args.pics}/slangs/'
            os.makedirs(langs_view, exist_ok=True)
            outname = f'{langs_view}{args.data}_{feat}:{args.y_column[0]}.png'

            plot_correlations_langs(my_df=_df, scale_them=strong_x + [args.y_column], x=feat, y=args.y_column,
                                    lang_map=SLANG_MAP, outpath=outname, show=args.verbose)

    end = time.time()
    print(f'\nTotal time: {((end - start) / 60):.2f} min')
