"""

USAGE:
python3 calculate_entropy.py --entropy_type user

"""

import argparse
import os
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from configs import MODEL


def calculate_valid_user_entropy(row, based_on=None):
    sample_list = row[based_on]

    # Count the frequency of each unique solution
    frequency_counts = Counter(sample_list)

    # Calculate the probabilities
    total_count = len(sample_list)
    probabilities = [count / total_count for count in frequency_counts.values()]

    # Calculate the entropy
    entropy = -sum(p * np.log2(p) for p in probabilities)

    return entropy


def make_dirs(logsto=None, resto=None, picsto=None):
    os.makedirs(resto, exist_ok=True)
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
    parser.add_argument('--intab', help="a reliable source of string cols after all fixes",
                        default=f'res/srp/{MODEL}/user_view/user_6579.tsv')
    parser.add_argument('--entropy_type', choices=['user', 'user_vars', 'normed_annotation'],
                        help="annotation or user variants?", default='user')
    parser.add_argument('--model', choices=['rugpt3large_based_on_gpt2', 'ruRoBERTa-large', 'xlm-roberta-base'],
                        default=MODEL)
    parser.add_argument('--res', default='res/entropy/')
    parser.add_argument('--pics', default='pics/entropy/')
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()
    start = time.time()

    make_dirs(logsto=args.logs, resto=args.res, picsto=args.pics)

    df = pd.read_csv(args.intab, sep='\t')
    print(df.shape)
    print(df.columns.tolist())
    # Create a copy of the 'user' column as 'user_vars'
    if args.entropy_type == 'user_vars':
        df['user_vars'] = df['user']
        df.loc[df['normed_annotation'].isin(['empty', 'noise']), 'user_vars'] = df['normed_annotation']
        print("Treating both `empty' and 'noise' as separate responses, using annotation tags")
    elif args.entropy_type == 'user':
        print("Treating both `empty' and 'noise' as None")
    else:
        print(set(df.normed_annotation.tolist()))
        print("Calculating entropy of annotations, i.e. solution types entropy instead of true MSU translation entropy")
    print(df.head())
    print(df.tail())

    collector = []
    for slang in df.language.unique():
        slang_df = df[df.language == slang]

        # Calculate entropy of valid translation solution by item and mean
        entropy_df = slang_df[['src', args.entropy_type, 'normed_annotation']]

        grouped_entropy_df = entropy_df.groupby('src', as_index=False).agg({args.entropy_type: lambda x: x.tolist(),
                                                                            'normed_annotation': lambda x: x.tolist()})
        grouped_entropy_df[f'{args.entropy_type}_entropy'] = grouped_entropy_df.apply(lambda x:
                                                                                      calculate_valid_user_entropy(x,
                                                                                                                   based_on=args.entropy_type),
                                                                                      axis=1)
        grouped_entropy_df['user_vars_num'] = grouped_entropy_df[args.entropy_type].apply(lambda x: len(set(x)))

        entropy_map = grouped_entropy_df.set_index('src')[f'{args.entropy_type}_entropy'].to_dict()
        num_map = grouped_entropy_df.set_index('src')['user_vars_num'].to_dict()

        lang_df = slang_df.copy()
        lang_df['entropy'] = slang_df['src'].map(entropy_map)
        lang_df['user_vars_num'] = slang_df['src'].map(num_map)

        collector.append(lang_df)

    enriched_df = pd.concat(collector, axis=0)
    print(enriched_df.head())

    enriched_df.to_csv(f'{args.res}source_phrase_entropies-on:{args.entropy_type}.tsv', sep='\t', index=False)

    end = time.time()
    print(f'\nTotal time: {((end - start) / 60):.2f} min')
