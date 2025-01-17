"""
25 Jul 2024
this is a variant of COMET calculation using evaluation utility (not in terminal), no need to save parallel sentences externally

pip install --upgrade pip
pip install evaluate comet_ml
huggingface-cli login
# or using an environment variable
huggingface-cli login --token $HUGGINGFACE_TOKEN
git config --global credential.helper store

python3 comet_evaluation.py --model Unbabel/wmt22-comet-da
python3 comet_evaluation.py --model Unbabel/wmt22-cometkiwi-da --qe
"""

from collections import defaultdict

import sys
import pandas as pd
import argparse
import os
import time
from datetime import datetime
import evaluate


# Function to write a list to a file
def write_list_to_file(list_to_write, filename):
    with open(filename, 'w') as file:
        for item in list_to_write:
            file.write(f"{item}\n")


def filter_out_user_noise(data=None):
    filtered = data[(data.normed_annotation != 'noise') & (data.normed_annotation != 'empty')]
    filtered['normed_user'] = filtered['normed_user'].str.lower()

    return filtered


def replace_substring_with_lit(row):
    if row['normed_gold'] in row['translated_sentence']:
        res = row['translated_sentence'].replace(row['normed_gold'], row['gpt4_literal'])
    else:
        print('This should happen! Replacing golds with lit ...')
        exit()
        if row['normed_gold'].capitalize() in row['translated_sentence'].capitalize():
            res = row['translated_sentence'].replace(row['normed_gold'].capitalize(), row['gpt4_literal'])
        else:
            # print(row['normed_gold'].value, row['translated_sentence'].value)
            # print('This should not happen!')
            res = row['translated_sentence'].replace(row['normed_gold'].lower(), row['gpt4_literal'])
    return res


def replace_substring_with_user(row):
    res = row['translated_sentence'].replace(row['normed_gold'], row['normed_user'])
    return res


def get_parallel_lists(my_df=None, tgt_name=None):  # get various types of parallel lists
    src_lst = my_df['sentence'].tolist()
    tgt_lst = my_df[f'{tgt_name}_translated_sentence'].tolist()
    ref_lst = my_df['translated_sentence'].tolist()

    ids_user = my_df['ID'].tolist()

    return src_lst, tgt_lst, ref_lst, ids_user


def get_comet_scores_evaluate(s=None, t=None, r=None, model=None, gpus=0):
    # List available models
    # comet_metric = evaluate.load('comet')
    # available_models = comet_metric.info['models']
    # print("Available COMET models:", available_models)
    # exit()
    comet_metric = evaluate.load('comet', model)
    results = comet_metric.compute(predictions=t, references=r, sources=s, gpus=gpus, progress_bar=True)
    score_list = results["scores"]

    return score_list


class Logger(object):
    def __init__(self, logfile=None):
        self.terminal = sys.stdout
        self.log = open(logfile, "w")

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


# Redirect stdout and stderr to the log file
class StdErrLogger(object):
    def __init__(self, logfile=None):
        self.terminal = sys.stderr
        self.log = open(logfile, "w")  # overwrite, don't "a" append

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, message):
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def make_dirs(logsto=None, res_to=None):
    os.makedirs(res_to, exist_ok=True)
    os.makedirs(logsto, exist_ok=True)
    current_datetime = datetime.utcnow()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H:%M')
    script_name = sys.argv[0].split("/")[-1].split(".")[0]

    log_file = f'{logsto}{formatted_datetime.split("_")[0]}_{script_name}.log'
    sys.stdout = Logger(logfile=log_file)
    print(f"\nRun date, UTC: {datetime.utcnow()}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")

    # Redirect sys.stderr to a file
    stderr_log_file = f'{logsto}stderr_{script_name}.log'
    sys.stderr = StdErrLogger(logfile=stderr_log_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate COMET qe and eval scores for tgt with replacement.')
    parser.add_argument('--intab', help="", default='data/normalised_translation_experiment_data.tsv')
    parser.add_argument('--model', choices=['Unbabel/wmt22-comet-da', 'Unbabel/wmt22-cometkiwi-da'],
                        default="Unbabel/wmt22-comet-da")
    parser.add_argument('--qe', action='store_true', help='pass this flag for quality estimation (no ref) models:'
                                                          ' (Unbabel/wmt22-cometkiwi-da, zwhe99/wmt21-comet-qe-da)')
    parser.add_argument('--eval_vs', choices=['ref=nllb', 'ref=gold'], default='ref=gold')
    parser.add_argument('--res', default='res/comet/')
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()

    start = time.time()

    make_dirs(logsto=args.logs, res_to=args.res)

    if args.qe:
        save_as = f'{args.res}qe_scores_{args.model.replace("Unbabel/", "")}.tsv'
    else:
        save_as = f'{args.res}eval_{args.eval_vs}_scores_{args.model.replace("Unbabel/", "")}.tsv'

    if os.path.exists(save_as):
        new_df0 = pd.read_csv(save_as, sep='\t')
        print(f'\n== translation quality scores from {args.model} are restored from file ==\n')
    else:
        print(f'\n== Generating {args.model} translation quality scores anew ... ==')
        df0 = pd.read_csv(args.intab, sep='\t')
        print(df0.columns.tolist())
        print(df0.shape)

        slangs_collector = []
        if args.qe:
            # sanity check
            if args.model in ['wmt20-comet-da', 'wmt21-comet-da', 'Unbabel/wmt22-comet-da']:
                print(f'{args.model} requires reference')
                exit()
            print('Running quality estimation metric (no ref) on nllb, ht-gold, ht-lit, ht-user ...')
        else:
            print(f'Running ref-based quality evaluation for nllb/gold, ht-lit, ht-user on {args.eval_vs} ...')

        for slang in df0.language.unique():
            slang_scores_collector = []
            this_lang = df0[df0.language == slang]
            print(f"\nSL: {slang}")
            this_lang = this_lang[['ID', 'language', 'normed_phrase', 'sentence', 'translated_sentence', 'normed_gold',
                                   'gpt4_literal', 'normed_user', 'normed_annotation']]
            # print(this_lang.head())
            if args.qe:
                my_tgts = ['gold', 'nllb', 'lit', 'user']
                meth_name = 'qe'
            else:
                if args.eval_vs == 'ref=nllb':
                    my_tgts = ['gold', 'lit', 'user']
                else:
                    my_tgts = ['nllb', 'lit', 'user']
                meth_name = 'eval'

            # _this_lang = this_lang.copy()
            for this_tgt in my_tgts:
                print(this_tgt)
                if this_tgt == 'lit':
                    _this_lang = this_lang.drop_duplicates(subset='sentence', keep='first')
                    # print(_this_lang.tail())
                    _df = _this_lang.copy()
                    _df[f'{this_tgt}_translated_sentence'] = _df.apply(replace_substring_with_lit, axis=1)

                elif this_tgt == 'user':
                    # get only the rows with meaningful users responses
                    _this_lang = filter_out_user_noise(data=this_lang)
                    # print(_this_lang)
                    _df = _this_lang.copy()
                    # replace gold translation with the user version
                    _df[f'{this_tgt}_translated_sentence'] = _df.apply(replace_substring_with_user, axis=1)

                elif this_tgt == 'nllb':
                    _this_lang = this_lang.drop_duplicates(subset='sentence', keep='first')
                    _df = _this_lang.copy()
                    # print(_df.columns.tolist())
                    _df[f'{this_tgt}_translated_sentence'] = _df['nllb_translated']
                else:  # gold
                    _this_lang = this_lang.drop_duplicates(subset='sentence', keep='first')
                    _df = _this_lang.copy()
                    _df[f'{this_tgt}_translated_sentence'] = _df['translated_sentence']

                src, tgt, ref, ids = get_parallel_lists(my_df=_df, tgt_name=this_tgt)

                comet_scores = get_comet_scores_evaluate(s=src, t=tgt, r=ref, model=args.model, gpus=0)
                # print(len(comet_scores), len(src), len(tgt))
                assert len(comet_scores) == len(src) == len(tgt), 'Huston, we have problems!'

                comet_scores_dict = defaultdict(list)
                colname = f'{meth_name}_{this_tgt}'
                res_lang = this_lang.copy()
                if this_tgt == 'user':
                    for my_id, score, tgt_sent in zip(ids, comet_scores, tgt):
                        comet_scores_dict[my_id].append(score)
                        comet_scores_dict[my_id].append(tgt_sent)
                    # print(comet_scores_dict)
                    # Create the new column using .apply() and a lambda function
                    res_lang[colname] = this_lang['ID'].map(lambda x: comet_scores_dict.get(x, [None])[0])
                    res_lang[f'{this_tgt}_translated_sentence'] = this_lang['ID'].map(lambda x: comet_scores_dict.get(x, [None, None])[1])
                else:
                    for ssent, score, tgt_sent in zip(src, comet_scores, tgt):
                        comet_scores_dict[ssent].append(score)
                        comet_scores_dict[ssent].append(tgt_sent)
                    res_lang[colname] = this_lang['sentence'].map(lambda x: comet_scores_dict.get(x, [None])[0])
                    res_lang[f'{this_tgt}_translated_sentence'] = this_lang['sentence'].map(lambda x: comet_scores_dict.get(x, [None, None])[1])
                    # print(res_lang.head())
                res_lang = res_lang.drop(['sentence', 'language', 'normed_phrase', 'translated_sentence', 'normed_user', 'nllb_translated',
                                          'normed_gold', 'gpt4_literal', 'normed_annotation'], axis=1).set_index('ID', drop=True)
                slang_scores_collector.append(res_lang)
            this_lang_scores = pd.concat(slang_scores_collector, axis=1)
            phrases_this_lang = this_lang[['ID', 'language', 'normed_phrase', 'normed_gold', 'gpt4_literal', 'normed_user',
                                           'normed_annotation', 'sentence']].set_index('ID', drop=True)
            phrases_this_lang_scores = pd.concat([phrases_this_lang, this_lang_scores], axis=1)
            slangs_collector.append(phrases_this_lang_scores)
        new_df0 = pd.concat(slangs_collector, axis=0)
        new_df0 = new_df0.reset_index()
        # Write the DataFrame to a TSV file
        new_df0.to_csv(save_as, sep="\t", index=False)

    print(new_df0.head())

    print(f'\nScores {new_df0.shape} are written to: {resto}')
    print(new_df0.columns.tolist())

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
