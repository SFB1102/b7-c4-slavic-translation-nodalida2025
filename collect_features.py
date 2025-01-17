"""
29 Sept 2024
predictors (classifiables):
--surprisals (src, lit, gold)
--cosines (cosine_src-gold, cosine_src-lit)
--distances (PWLD)

user-based explanatory variables:
--surprisals (user)
--cosine (cosine_src-user)
--translation quality (no-ref is more fair than ref-based, probably)
--translation entropy of a source phrase (based on the distribution of user responses)

surprisal and entropies need to be stored by model, in {model}.tsv files
distances, entropies, quality values can be in a separate one file

USAGE:
python3 collect_features.py --model ruRoBERTa-large --how uniq --entropy_type user

"""

import argparse
import os
import sys
import time
from datetime import datetime
from collections import Counter

import pandas as pd

from configs import MODEL, ANNO_ORDER


def get_score(list1, list2):
    scores = [a * b for a, b in zip(list1, list2)]
    score = sum(scores)

    return score


def relative_frequencies_ordered(my_list, order=None):
    item_counts = Counter(my_list)
    total_items = len(my_list)
    relative_freqs = [item_counts[item] / total_items for item in order]

    return relative_freqs


def loop_langs_and_serialise_dist(matrix_df, uniq_df):
    collector = []
    dist_columns = ['pwld_original_lit', 'pwld_gold_lit', 'pwld_gold_original']
    # print(dist_columns)
    for lang in matrix_df.language.unique():
        lang_collector = []
        this_matrix_df = matrix_df[matrix_df['language'] == lang]
        this_filler_data = uniq_df[uniq_df['language'] == lang]

        for dist in dist_columns:
            this_dist_map = this_filler_data.set_index('phrase')[dist].to_dict()
            # print(*(f"{k}: {v}" for k, v in list(this_dist_map.items())[:2]), sep='\n')

            # this_matrix_df = this_matrix_df.assign(**{dist: this_matrix_df['src'].map(this_dist_map)})
            temp_matrix_df = this_matrix_df.copy()
            # the columns containing original MSU in the five slavic languages are called src anf phrase
            # in different parts of the project for historical reasons
            temp_matrix_df[dist] = this_matrix_df['src'].map(this_dist_map)
            temp_matrix_df = temp_matrix_df.set_index('ID')
            temp_matrix_df = temp_matrix_df[[dist]]
            lang_collector.append(temp_matrix_df)
        this_lang_dist_df = pd.concat(lang_collector, axis=1)
        collector.append(this_lang_dist_df)

    matrix_redone = pd.concat(collector, axis=0)

    return matrix_redone


def make_dirs(logsto=None, resto=None, subdir=None):
    res_to = f'{resto}{subdir}/'
    os.makedirs(res_to, exist_ok=True)

    os.makedirs(logsto, exist_ok=True)
    current_datetime = datetime.utcnow()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H:%M')
    script_name = sys.argv[0].split("/")[-1].split(".")[0]
    log_file = f'{logsto}{formatted_datetime.split("_")[0]}_{script_name}.log'
    sys.stdout = Logger(logfile=log_file)
    print(f"\nRun date, UTC: {datetime.utcnow()}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")

    return res_to


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
    parser.add_argument('--indir', help="", required=False, default='data/deen/')
    parser.add_argument('--model', choices=['rugpt3large_based_on_gpt2', 'ruRoBERTa-large', 'xlm-roberta-base'],
                        default=MODEL)
    parser.add_argument('--entropy_type', choices=['user', 'user_vars', 'normed_annotation'],
                        help="annotation or user variants?", default='user')
    parser.add_argument('--how', choices=['uniq', 'user'], default='user')
    parser.add_argument('--res', default='data/final_wide_tables/')
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()
    start = time.time()

    resto = make_dirs(logsto=args.logs, resto=args.res, subdir=args.how)
    resto_user = resto.replace("uniq", "user")
    os.makedirs(resto_user, exist_ok=True)

    # quality
    qu_collect = []
    for qu_path in ['res/comet/qe_scores_wmt22-cometkiwi-da.tsv', 'res/comet/eval_ref=gold_scores_wmt22-comet-da.tsv']:
        qu_df = pd.read_csv(qu_path, sep='\t').set_index('ID')
        qu_collect.append(qu_df)
    qu_both = pd.concat(qu_collect, axis=1).reset_index()
    keep_columns = ['ID', 'qe_gold', 'eval_lit', 'qe_lit', 'eval_user', 'qe_user']
    quality_values = qu_both[keep_columns].set_index('ID')

    # entropy
    ent_path = f'res/entropy/source_phrase_entropies-on:{args.entropy_type}.tsv'
    ent_df = pd.read_csv(ent_path, sep='\t')
    keep_columns = ['ID', 'entropy', 'user_vars_num']

    entropy_values = ent_df[keep_columns].set_index('ID')

    # surprisal-cosines by model
    indices_df = pd.read_csv(f'res/emb/{args.model}/cosines_{args.model}.tsv', sep='\t')

    keep_columns = ['ID', 'language', 'src', 'category', 'normed_annotation', 'surprisal_src', 'surprisal_lit',
                    'surprisal_gold', 'surprisal_src_sent', 'surprisal_lit_sent', 'surprisal_gold_sent',
                    'cosine_src-gold', 'cosine_src-lit',
                    'surprisal_user', 'surprisal_user_sent', 'cosine_src-user']
    transformer_values = indices_df[keep_columns].set_index('ID')

    string_df = indices_df[['ID', 'language', 'category', 'normed_annotation', 'src', 'gold', 'lit', 'user', 'src_sent',
                            'gold_sent', 'lit_sent', 'user_sent']]
    if args.how == 'uniq':
        str_outf = f'{resto}uniq_string_columns.tsv'
        uniq_str_df = string_df.copy()
        uniq_str_df['item_ID'] = string_df['ID'].apply(lambda x: x.split(':')[0])
        uniq_str_df['response_probability'] = uniq_str_df['normed_annotation'].copy()

        cols2first = ['item_ID', 'language', 'src', 'category', 'gold', 'lit']
        agg_dict_first = {col: 'first' for col in cols2first}

        cols2list = ['normed_annotation', 'user']
        agg_dict_list = {col: lambda x: x.tolist() for col in cols2list}

        agg_dict_ratio = {'response_probability': lambda x: relative_frequencies_ordered(x.tolist(), order=ANNO_ORDER)}

        agg_dict = {**agg_dict_first, **agg_dict_list, **agg_dict_ratio}
        string_df = uniq_str_df.groupby('item_ID').agg(agg_dict)

        # add tot_responces and responces_dict columns based on lists in user column
        string_df['tot_responses'] = string_df['user'].apply(len)

        # ordered freq dict
        string_df['responses_dict'] = string_df['user'].apply(
            lambda x: dict(sorted(Counter(x).items(), key=lambda item: item[1], reverse=True)))
        # string_df['responses_dict'] = string_df['user'].apply(lambda x: {i: val for i, val in enumerate(x)})
    else:
        str_outf = f'{resto.replace("uniq", "user")}user_string_columns.tsv'

    string_df.to_csv(str_outf, sep='\t', index=False)
    print('String features table:')
    print(string_df.columns.tolist())
    print(string_df.head())

    # distances
    dist_path = 'data/input/data_with_gptlit_translit_distances.csv'
    dist_from = pd.read_csv(dist_path, sep=',')
    # ['ID', 'pwld_original_lit', 'pwld_gold_lit', 'pwld_gold_original']
    distances_values = loop_langs_and_serialise_dist(ent_df, dist_from)

    print(len(transformer_values), len(distances_values), len(entropy_values), len(quality_values))
    assert len(transformer_values) == len(distances_values) == len(entropy_values) == len(quality_values), 'Kwa-kwa-kwa'

    all_features = pd.concat([transformer_values, distances_values, entropy_values, quality_values],
                             axis=1).reset_index()

    print('All features table:')
    print(all_features.columns.tolist())
    print(all_features.shape)
    all_outf = f'{resto_user}features{all_features.shape[1] - 5}_{args.model}.tsv'
    all_features.to_csv(all_outf, sep='\t', index=False)

    feature_map = {'predictors': ['surprisal_src', 'surprisal_lit', 'surprisal_gold',
                                  'surprisal_src_sent', 'surprisal_lit_sent', 'surprisal_gold_sent',
                                  'cosine_src-gold', 'cosine_src-lit',
                                  'pwld_original_lit', 'pwld_gold_lit', 'pwld_gold_original',
                                  'qe_gold', 'eval_lit', 'qe_lit',
                                  'entropy', 'user_vars_num'],
                   'descriptors': ['surprisal_user', 'surprisal_user_sent', 'cosine_src-user',
                                   'eval_user', 'qe_user']}

    for name, feats in feature_map.items():
        keep_columns = ['ID', 'language', 'src', 'category', 'normed_annotation'] + feats
        this_feature_set = all_features[keep_columns]
        outf = f'{resto.replace("uniq", "user")}{name}_{args.model}.tsv'
        this_feature_set.to_csv(outf, sep='\t', index=False)
        print(f'{name.upper()} features table:')
        print(this_feature_set.columns.tolist())
        print(this_feature_set.shape)

    if args.how == 'uniq':
        uniq_df = all_features.copy()
        uniq_df['item_ID'] = all_features['ID'].apply(lambda x: x.split(':')[0])

        cols2first = ['item_ID', 'language', 'src', 'category', 'surprisal_src', 'surprisal_lit', 'surprisal_gold',
                      'surprisal_src_sent', 'surprisal_lit_sent', 'surprisal_gold_sent',
                      'cosine_src-gold', 'cosine_src-lit',
                      'pwld_original_lit', 'pwld_gold_lit', 'pwld_gold_original',
                      'qe_gold', 'eval_lit', 'qe_lit',
                      'entropy', 'user_vars_num']
        agg_dict_first = {col: 'first' for col in cols2first}

        cols2mean = ['surprisal_user', 'surprisal_user_sent', 'cosine_src-user',
                     'eval_user', 'qe_user']
        agg_dict_mean = {col: 'mean' for col in cols2mean}

        # anno_order = ['correct', 'fluent_literal', 'paraphrase', 'awkward_literal', 'fantasy', 'noise', 'empty']
        agg_dict_ratio = {'normed_annotation': lambda x: relative_frequencies_ordered(x.tolist(), order=ANNO_ORDER)}

        agg_dict = {**agg_dict_first, **agg_dict_mean, **agg_dict_ratio}

        groupped_uniq_df = uniq_df.groupby('item_ID').agg(agg_dict)

        groupped_uniq_df = groupped_uniq_df.rename(columns={'normed_annotation': 'response_probability'})

        # response_weights = {'correct': 7,
        #                     'fluent_literal': 6,
        #                     'paraphrase': 5,
        #                     'awkward_literal': 4,
        #                     'fantasy': 2,
        #                     'noise': 0,
        #                     'empty': 0}
        response_weights = [7, 6, 5, 4, 2, 0, 0]

        uniq_df_score = groupped_uniq_df.copy()
        uniq_df_score['intelligibility'] = groupped_uniq_df['response_probability'].apply(
            lambda x: get_score(x, response_weights))

        uniq_outf = f'{resto}scores_ratios_responses_{args.model}.tsv'
        uniq_df_score.to_csv(uniq_outf, sep='\t', index=False)
        print(f'{args.how.upper()} features table:')
        print(uniq_df_score.columns.tolist())
        print(uniq_df_score.head())
        print(uniq_df_score.shape)

    end = time.time()
    print(f'\nTotal time: {((end - start) / 60):.2f} min')
