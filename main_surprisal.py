"""
29 Jul 2024

requires the output of comet_evaluation.py

get the surprisal for literal and user variants of the focused MWE in the context of the provided reference sentences
in the target language (Russian) using a dedicated Russian pre-trained model

plot the distributions of the surprisals by SL and (a) PoS of the item, (b) translation solution

(a) source or (b) target sentence from ruTransformer between
-- exp1: source/gold_translation (baseline) and user translation solutions regardless type of solution and type of item
-- exp2: for each SL, source/gold_translation (baseline) and types of user translation solutions (literal, correct, fantasy)
-- exp3: for each SL, between source (or gold translation) and translations by class of items (particles, conj, etc)

python3 main_surprisal.py --model ruRoBERTa-large --exp slangs
python3 main_surprisal.py --model xlm-roberta-base --exp slangs
"""
import pickle
import sys
from collections import defaultdict
import ast

import numpy as np
import pandas as pd
import argparse
import os
import time
from datetime import datetime

import transformers
from tqdm import tqdm

import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import unicodedata

import warnings
from transformers import RobertaForMaskedLM, RobertaTokenizer, AutoModelForMaskedLM
from transformers import BertTokenizer, BertForMaskedLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModel, AutoConfig
from unidecode import unidecode
import re
from minicons import scorer
from configs import SLANG_MAP, PREDICTORS, DESCRIPTORS, LANGUAGE_ORDER, POS_ORDER, ANNO_ORDER, MODEL


warnings.filterwarnings("ignore")


def aggregate_subwords(words, subwords, values):
    aggregated_words = []
    aggregated_values = []

    word_idx = 0
    current_word = ""
    current_value_sum = 0.0

    for subword, value in zip(subwords, values):
        current_word += subword  # Continue building the word by adding the subword
        current_value_sum += value  # Sum the value corresponding to the subword

        # Once the built word matches the current word in the words list
        try:
            if current_word == words[word_idx]:
                aggregated_words.append(current_word)  # Add the full word
                aggregated_values.append(current_value_sum)  # Add the sum of values for the word

                # Reset for the next word
                current_word = ""
                current_value_sum = 0.0
                word_idx += 1  # Move to the next word
        except IndexError:
            print(word_idx)
            print(current_word)
            print(subwords)
            print(words)
            exit()

    if not len(aggregated_words) == len(aggregated_values) == len(words):
        print()
        print(subwords)
        print(words)
        print(aggregated_words)
        print(len(aggregated_words), len(aggregated_values), len(words))
        # manually fixed:
        # in BG user: 'с', 'бронзовым', 'набалдашником', 'в', '\u200b\u200b', 'форме', 'сфинкса'
        print('Kwa-Kwa_Kwa!')
        exit()
    else:
        return aggregated_words, aggregated_values


def calculate_surprisal(sentence, form):
    if args.model == 'xlm-roberta-base':
        # dirty tokenisation fixes
        sentence = sentence.replace('…', '').replace('.', '').replace('!', '').replace('%', '').replace('-', ' - ')
        sentence = sentence.replace('(', '').replace(')', '').replace('«', '').replace('»', '').replace('–', '')
    elif args.model == 'rugpt3large_based_on_gpt2':
        sentence = sentence.replace('…', '').replace('.', '').replace('!', '').replace('%', '').replace('-', ' - ')
        sentence = sentence.replace('(', '').replace(')', '').replace('«', '').replace('»', '').replace('–', '')
        sentence = sentence.replace('―', '').replace('?..', '?')
    else:  # ruRoberta seems to be the most robust
        pass

    try:
        # base-2 logarithm returns the largest values among base-10, base-e (2.71828) and base-2
        # ('ад', 'ным', 'словам', ',', 'пад', 'м', 'ін', 'ск', 'ам', 'е', 'с', 'ць', 'такая', 'спец', 'ыя', 'ль', 'ная')
        if 'gpt' in args.model:
            sent_sub_surprisal = MODEL.token_score(sentence, surprisal=True, base_two=True)[0]
        else:
            sent_sub_surprisal = MODEL.token_score(sentence, surprisal=True, base_two=True, PLL_metric=METRIC)[0]

    except IndexError:
        print(f"Error with input: {sentence}")
        return [None, None]

    sent_subtoks, sent_vals = zip(*sent_sub_surprisal)
    # Remove empty strings and corresponding values using list comprehension
    sent_subtoks = [s for s in sent_subtoks if s != '']
    sent_vals = [v for s, v in zip(sent_subtoks, sent_vals) if s != '']

    if args.model == 'xlm-roberta-base':
        # more dirty tokenisation fixes
        # treat 4,7 as one token, and zničen 21. července 356 př.n.l . v dusledku
        raw_tokens = re.findall(r'\d+(?:[.,]\d+)?\.?|\w+(?:\.\w+)*\b(?:\.)?|[^\w\s](?<!\.)', sentence, re.UNICODE)
        # raw_tokens = re.findall(r'\d+(?:[.,]\d+)?|\w+|[^\w\s]+', sentence, re.UNICODE)
    else:
        # Regex to capture words and punctuation sequences
        raw_tokens = re.findall(r'\w+|[^\w\s]+', sentence, re.UNICODE)
    # print(args.model)
    # print(sentence)
    # print(sent_subtoks)
    # print(len(sent_subtoks))
    # exit()

    words_lst, word_surprisals = aggregate_subwords(raw_tokens, sent_subtoks, sent_vals)

    # applying the same preprocessing to items seems to be an overkill, but I realise this approach is fragile
    form_toks = re.findall(r'\w+|[^\w\s]+', form, re.UNICODE)
    extracted_form = None
    form_srp = None
    try:
        # Extract indices of all occurrences of the 1st item
        possible_start_indices = [i for i, x in enumerate(words_lst) if x == form_toks[0]]
        # start_idx = words_lst.index(form_toks[0])
        if len(possible_start_indices) != 1:
            if len(form_toks) > 1:  # if MSU is multiword! not the case in BG досега and its' user's variant почему
                for start_idx in possible_start_indices:
                    next_idx = start_idx + 1
                    if words_lst[next_idx] == form_toks[1]:
                        # print(words_lst[next_idx + 1])
                        # print(form_toks)
                        if '-' in words_lst[next_idx]:  # най - лошото instead of най - малкото in BG
                            if words_lst[next_idx + 1] == form_toks[2]:
                                # print(words_lst[next_idx + 1])
                                # print(form_toks)
                                end_idx = start_idx + len(form_toks)
                                form_srp = sum(sent_vals[start_idx:end_idx])
                                extracted_form = ' '.join(words_lst[start_idx:end_idx])
                                break
                        else:
                            end_idx = start_idx + len(form_toks)
                            form_srp = sum(sent_vals[start_idx:end_idx])
                            extracted_form = ' '.join(words_lst[start_idx:end_idx])
                            break
                    else:
                        continue
            else:
                # досега is 2nd почему
                # куда исчез, почему – почему загадка. а команда шикина разбрелась кто куда.
                oneword_msu_last_idx = len(words_lst) - 1 - words_lst[::-1].index(form_toks[0])
                form_srp = sent_vals[oneword_msu_last_idx]
                extracted_form = words_lst[oneword_msu_last_idx]
        elif len(possible_start_indices) == 0:
            print('First component of the MSU is not in sentence? Whaaaat??')
            print(words_lst)
            print(form_toks)
            exit()
        else:
            start_idx = words_lst.index(form_toks[0])
            end_idx = start_idx + len(form_toks)
            form_srp = sum(sent_vals[start_idx:end_idx])
            extracted_form = ' '.join(words_lst[start_idx:end_idx])
    except ValueError:
        print(f'I cannot find "{form}" in "{sentence}".')
        print(words_lst)
        print(form_toks)
        exit()
    except IndexError:
        print('Index error!')
        print(words_lst)
        print(form_toks)

    if ' '.join(form_toks) != extracted_form:
        print('\nYou extracted surprisal for something else!')
        print(' '.join(form_toks))
        print(extracted_form)
        print(sentence)
        print(words_lst)
        exit()

    return form_srp, np.mean(list(sent_vals))


def get_model_srp(_data, paired_cols=None):
    updaded_data = _data.copy()
    err_dict = defaultdict(int)
    for cols in paired_cols:
        av_sent_srp_lst = []
        phrase_srp_lst = []
        form_col_name, sent_col_name = cols
        print(form_col_name)
        err_count = 0
        for _, row in tqdm(_data.iterrows(), position=0, leave=True):
            av_sent_srp_val = None
            phrase_srp_val = None
            if sent_col_name in row and isinstance(row[sent_col_name], str):
                if form_col_name in row and isinstance(row[form_col_name], str):
                    phrase_srp_val, av_sent_srp_val = calculate_surprisal(row[sent_col_name], row[form_col_name])
                    if not phrase_srp_val:
                        err_count += 1
            av_sent_srp_lst.append(av_sent_srp_val)
            phrase_srp_lst.append(phrase_srp_val)
        updaded_data.insert(1, f'surprisal_{form_col_name}', phrase_srp_lst)
        updaded_data.insert(1, f'surprisal_{sent_col_name}', av_sent_srp_lst)
        err_dict[form_col_name] = err_count

    err_df = pd.DataFrame(err_dict, index=[0])

    return updaded_data, err_df


def boxplot_srp(datas=None, my_exp=None, colours=None, lang_order=None, subs_order=None, outdir=None, outf=None,
                text=None, lose_outliers=None):
    # my_hue=args.exp: 'slangs', 'category', 'normed_annotation'
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.set_context('paper')
    # Set the default font properties
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 14

    if my_exp == 'slangs':
        # reshape -- wide to long! and two table sizes 60 and user
        collector = []
        for name, _df in datas.items():
            if name == 'uniq':
                plottable = _df[['language', 'surprisal_src', 'surprisal_lit', 'surprisal_gold',
                                 'src', 'lit', 'gold']]
                # Melt the dataframe to long format for surprisal values
                print(plottable.head())

                long_srp_df = pd.melt(plottable, id_vars=['language'],
                                      value_vars=['surprisal_src', 'surprisal_lit', 'surprisal_gold'],
                                      var_name='itm_type', value_name='surprisal')
                print(long_srp_df.head())

                # Replace the 'itm_type' values to match the desired output
                long_srp_df['itm_type'] = long_srp_df['itm_type'].str.replace('surprisal_', '')

                collector.append(long_srp_df)
                print("Uniq df")
                print(long_srp_df.head())
                print(long_srp_df.tail())
                print(long_srp_df.shape)
            else:  # user
                plottable_user = _df[['language', 'surprisal_user', 'user']]
                plottable_user = plottable_user.rename(columns={'surprisal_user': 'surprisal',
                                                                'user': 'item'})
                plottable_user.insert(1, 'itm_type', 'user')

                collector.append(plottable_user)

        exp_plot_me = pd.concat(collector, axis=0)

        print(set(exp_plot_me['itm_type'].tolist()))
        print(exp_plot_me.tail())
        print(exp_plot_me.columns.tolist())
        print(exp_plot_me.shape)

        if lose_outliers:
            THRES = 10
            # filter out surprisals over THRES in _srp columns
            exp_plot_me = exp_plot_me[exp_plot_me['surprisal'] <= THRES]
            print(exp_plot_me.shape)

        # do I have NaN in srp?? where? why?
        nan_count_srp = exp_plot_me['surprisal'].isna().sum()
        print(f"Number of NaN values in surprisal: {nan_count_srp}")

        sns.boxplot(
            data=exp_plot_me,
            x='language',
            y='surprisal',
            hue='itm_type',
            hue_order=subs_order,
            order=lang_order,
            palette=colours,
            ax=ax
        )
        plt.legend(title='Translation status:', bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=12,
                   title_fontsize=12)
        plt.tight_layout()
        # replace src to source and lit to GPT4-literal
        custom_labels = {
            'gold': 'gold translation',
            'lit': 'GPT4-literal',
            'src': 'source phrase',
            'user': 'user response',
        }
        handles, labels = plt.gca().get_legend_handles_labels()
        custom_labels = [custom_labels[label] for label in labels]
        # Set the customized legend
        plt.legend(handles, custom_labels, title='Translation status:', bbox_to_anchor=(1.05, 1.0), loc='upper left',
                   fontsize=12, title_fontsize=12)

    else:  # 'normed_annotation' or category:
        exp_plot_me = datas['user']
        sns.boxplot(
            data=exp_plot_me,
            x='language',
            y='surprisal_user',
            hue=my_exp,
            hue_order=subs_order,
            order=lang_order,
            palette=colours,
            ax=ax
        )

        plt.legend(title='', bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=12, title_fontsize=12)
    plt.xticks(fontsize=12)  # Adjust the font size of x-axis tick labels
    plt.yticks(fontsize=12)
    # Add labels and title
    plt.xlabel('')
    plt.ylabel('Surprisal of the user response', fontsize=12)
    plt.title('')  # Caption for the Figure
    plt.tight_layout()
    if text:
        # Adding an info box
        plt.text(
            x=0.01, y=0.09, s=text,  # Position of the info box (x-axis within the plot area)
            transform=plt.gca().transAxes,  # Use Axes coordinates
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')  # Background and border
        )

    # Show the plot
    plt.savefig(outdir + outf)
    # plt.show()
    tables_out = f'{outdir}tables/'
    os.makedirs(tables_out, exist_ok=True)
    exp_plot_me.to_csv(f'{tables_out}long_{outf.replace(".png", ".tsv")}', sep='\t',
                       index=False)


def clean_text(text):
    return ''.join([char if char.isprintable() else '' for char in text])


def save(filename=None, vect=None):
    pickle.dump(vect, open(filename, 'wb'), pickle.HIGHEST_PROTOCOL)


def restore(filename=None):
    vect = pickle.load(open(filename, 'rb'))
    return vect


def make_dirs(logsto=None, pics=None, res=None, mmodel=None):
    logsto = f'{logsto}{mmodel}/'
    os.makedirs(logsto, exist_ok=True)
    res_to = f'{res}{mmodel}/'
    os.makedirs(res_to, exist_ok=True)
    pics_to = f'{pics}{mmodel}/'
    os.makedirs(pics_to, exist_ok=True)

    current_datetime = datetime.utcnow()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H:%M')
    script_name = sys.argv[0].split("/")[-1].split(".")[0]
    log_file = f'{logsto}{mmodel}_{formatted_datetime.split("_")[0]}_{script_name}.log'
    sys.stdout = Logger(logfile=log_file)
    print(f"\nRun date, UTC: {datetime.utcnow()}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")
    return temp_to, res_to, pics_to


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
    parser = argparse.ArgumentParser(description='calculate surprisals')
    parser.add_argument('--intab', help="use QE res as textual input as it has all 5 versions of the sent",
                        default='res/comet/qe_scores_wmt22-cometkiwi-da.tsv')
    parser.add_argument('--cats_from', help="main df",
                        default='data/normalised_translation_experiment_data.tsv')
    parser.add_argument('--exp', choices=['slangs', 'category', 'normed_annotation'], required=True)
    parser.add_argument('--model', choices=['rugpt3large_based_on_gpt2', 'ruRoBERTa-large', 'xlm-roberta-base'],
                        default=MODEL)
    parser.add_argument('--res', default=f'res/srp/')
    parser.add_argument('--pics', default="pics/srp/")
    parser.add_argument('--plotting', type=int, default=0)
    parser.add_argument('--logs', default='logs/srp/')

    args = parser.parse_args()

    start = time.time()
    resto, picsto = make_dirs(logsto=args.logs, res=args.res, pics=args.pics, mmodel=args.model)

    METRIC = "within_word_l2r"  # "original"
    if args.model == 'xlm-roberta-base':
        MODEL = scorer.MaskedLMScorer(args.model, 'cpu')
    else:
        if 'gpt' in args.model:
            MODEL = scorer.IncrementalLMScorer(f'ai-forever/{args.model}', 'cpu')
        else:
            MODEL = scorer.MaskedLMScorer(f'ai-forever/{args.model}', 'cpu')  # ruRoBERTa-large

    DIACRITIC_LANGUAGES = ['PL', 'CS']
    # df with translation variants in context of "translated_sentence"
    df0 = pd.read_csv(args.intab, sep='\t').set_index('ID')
    print(df0.tail())

    # add item type from an external df
    cats = pd.read_csv(args.cats_from, sep='\t', usecols=['ID', 'category']).set_index('ID')
    print(cats.tail())
    df1 = pd.concat([df0, cats], axis=1)
    df1 = df1.reset_index()

    # unify columns names
    df1 = df1.rename(
        columns={'normed_phrase': 'src', 'normed_gold': 'gold', 'gpt4_literal': 'lit', 'normed_user': 'user',
                 'sentence': 'src_sent', 'gold_translated_sentence': 'gold_sent',
                 'lit_translated_sentence': 'lit_sent', 'user_translated_sentence': 'user_sent'})
    # keep two dfs: for about 60 itms by lang and for all valid user inputs
    uniq_df = df1[['ID', 'language', 'category', 'src', 'src_sent', 'gold', 'gold_sent', 'lit', 'lit_sent']]

    uniq_df = uniq_df.drop_duplicates(subset='src_sent', keep='first')

    # get a df with more numerous user columns: keep dups if any!
    user_df = df1[['ID', 'language', 'category', 'normed_annotation', 'user', 'user_sent']]

    # 'user_translated_sentence' has only valid user variants (empty and noise is filtered out by COMET script)
    user_df = user_df.dropna(subset=['user_sent'])  # limit user df to non-empty response
    srp_uniq_save_as = f'{resto}srp_uniq-src-gold-lit_df_pos_{args.model}.pkl'
    srp_user_save_as = f'{resto}srp_user-df_anno+pos_{args.model}.pkl'

    if os.path.exists(srp_uniq_save_as):
        uniq_slangs_df = restore(srp_uniq_save_as)
        user_slangs_df = restore(srp_user_save_as)
        print(f'\n== Tables with surprisals from {args.model} are restored from pickle ==')
    else:
        print(f'\n== Calculating surprisals from {args.model} model anew ==')

        uniq_slangs_collector = []
        user_slangs_collector = []
        tot_err_dict = defaultdict()
        err_collector = []
        print(uniq_df.language.unique())

        for slang in uniq_df.language.unique():
            print(slang)
            lang_srp_uniq_save_as = srp_uniq_save_as.replace('res/', 'temp/').replace('srp_', f'{slang}_srp_')
            lang_srp_user_save_as = srp_user_save_as.replace('res/', 'temp/').replace('srp_', f'{slang}_srp_')
            lang_srp_errors_save_as = f'temp/srp/{args.model}/srp_errors_{args.model}.pkl'
            try:
                uniq_enriched_slang = restore(lang_srp_uniq_save_as)
                user_enriched_slang = restore(lang_srp_user_save_as)
                err_in_slang_phrases = restore(lang_srp_errors_save_as)
                print(f'\n== Dicts with surprisals for f{slang.upper()} are restored from pickle ==')
            except FileNotFoundError:
                print(f'\n== Calculating surprisals from {args.model} model anew ==')
                this_uniq_slang = uniq_df[uniq_df.language == slang]

                if slang in DIACRITIC_LANGUAGES:
                    for col in ["src", "src_sent"]:
                        this_uniq_slang[col] = this_uniq_slang[col].apply(unidecode)
                # є problem in єдиної: 'яно', 'ї', 'спор', 'уди', ',', '�', '�', 'дино', 'ї', 'кам', "'",
                if slang == 'UK':
                    for col in ["src", "src_sent"]:
                        # I tried normalize, explicitly handle encoding conversion, clean_text
                        this_uniq_slang[col] = this_uniq_slang[col].apply(lambda x: x.replace('є', 'е'))
                uniq_cols = [['src', 'src_sent'], ['gold', 'gold_sent'], ['lit', 'lit_sent']]
                # uniq_meta = ['ID', 'category', 'language']

                uniq_enriched_slang, uniq_srp_err_report = get_model_srp(this_uniq_slang, paired_cols=uniq_cols)
                # keep only meaningful columns: surprisal_ columns have single values:
                # AvS for sentences (do I even need this?), summed surprisal for words in phrases
                uniq_enriched_slang = uniq_enriched_slang[['ID', 'category', 'language',
                                                           'src', 'surprisal_src',
                                                           'src_sent', 'surprisal_src_sent',
                                                           'gold', 'surprisal_gold',
                                                           'gold_sent', 'surprisal_gold_sent',
                                                           'lit', 'surprisal_lit',
                                                           'lit_sent', 'surprisal_lit_sent']]

                this_user_slang = user_df[user_df.language == slang]
                user_cols = ['user', 'user_sent']
                user_meta = ['ID', 'category', 'language', 'normed_annotation']
                user_enriched_slang, user_srp_err_report = get_model_srp(this_user_slang, paired_cols=[user_cols])
                user_enriched_slang = user_enriched_slang[['ID', 'category', 'language', 'normed_annotation',
                                                           'user', 'surprisal_user',
                                                           'user_sent', 'surprisal_user_sent']]
                err_in_slang_phrases = pd.concat([uniq_srp_err_report, user_srp_err_report], axis=1)
                err_in_slang_phrases.insert(0, 'slang', slang)

                # pickling output for each language
                save(filename=lang_srp_uniq_save_as, vect=uniq_enriched_slang)
                save(filename=lang_srp_user_save_as, vect=user_enriched_slang)
                save(filename=lang_srp_errors_save_as, vect=err_in_slang_phrases)

            uniq_slangs_collector.append(uniq_enriched_slang)
            user_slangs_collector.append(user_enriched_slang)

            err_collector.append(err_in_slang_phrases)
            print('\nUNIQ (src, gold, lit) data with surprisals:')
            print(uniq_enriched_slang.columns.tolist())
            print(uniq_enriched_slang.shape)
            print('\nUSER data with surprisals:')
            print(user_enriched_slang.columns.tolist())
            print(user_enriched_slang.shape)
            print(f'*** {slang}: Surprisals for src and all translations (gold, lit, user) are ready. ***\n')

        uniq_slangs_df = pd.concat(uniq_slangs_collector, axis=0)
        user_slangs_df = pd.concat(user_slangs_collector, axis=0)

        final_msg = "== Tables with srp pickled ==" \
                    f"\n\tShape: {uniq_slangs_df.shape}, Location: {srp_uniq_save_as}" \
                    f"\n\tShape: {user_slangs_df.shape}, Location: {srp_user_save_as}" \
                    f"\n\tColumns: {uniq_slangs_df.columns.tolist()}"
        print(final_msg)

        srp_err_df = pd.concat(err_collector, axis=0)
        print(f'\n** {args.model} ** Errors in surprisal inference:')
        print(srp_err_df)

        # pickle to avoid learning vectors on each run with the same input
        save(filename=srp_uniq_save_as, vect=uniq_slangs_df)
        save(filename=srp_user_save_as, vect=user_slangs_df)

        vec_tot = time.time()
        print(f"\nTotal calculation time: {round((vec_tot - start) / 60, 2)} min")

    print(f'Uniq: {uniq_slangs_df.shape}')
    print(uniq_slangs_df.columns.tolist())
    print(f'User: {user_slangs_df.shape}')
    print(user_slangs_df.columns.tolist())

    uniq_slangs_df.to_csv(f'{resto}wide_srp-{args.model}_src-lit-gold.tsv', sep='\t',
                          index=False)
    user_slangs_df.to_csv(f'{resto}wide_srp-{args.model}_user.tsv', sep='\t', index=False)

    print(f'Surprisal for uniq sents for each MSU (src, gold, lit) and for multiple user responces are saved to {resto}wide...')

    #

    if args.plotting:
        # slang_map = {'CS': 'Czech', 'PL': 'Polish', 'BG': 'Bulgarian', 'UK': 'Ukrainian', 'BE': 'Belarusian'}
        uniq_slangs_df['language'] = uniq_slangs_df['language'].replace(SLANG_MAP)
        user_slangs_df['language'] = user_slangs_df['language'].replace(SLANG_MAP)

        # Define a custom color palette using the muted style
        # 10 colours https://seaborn.pydata.org/generated/seaborn.color_palette.html
        if args.exp == 'normed_annotation':
            cmap = plt.get_cmap('RdYlGn')
            # Generate 7 distinct colors from the colormap
            base_palette = [cmap(i) for i in range(0, 256, 256 // 7)][::-1]
            my_palette = {
                'correct': base_palette[0],
                'fluent_literal': base_palette[1],
                'paraphrase': base_palette[2],
                'awkward_literal': base_palette[3],
                'fantasy': base_palette[4],
                'noise': base_palette[5],
                'empty': base_palette[6]
            }

            my_subs_order = ['correct', 'fluent_literal', 'paraphrase', 'awkward_literal', 'fantasy']
            plot_me = {'user': user_slangs_df}
        elif args.exp == 'category':
            # 'parenth', 'adv_pred', 'conj', 'prep', 'part', 'other'
            base_palette = sns.color_palette("muted")
            my_palette = {
                'parenth': base_palette[9],  # green-like color from muted palette
                'conj': base_palette[8],  # orange-like color from muted palette
                'adv_pred': base_palette[7],  # red-like color from muted palette
                'part': base_palette[6],  # purple-like color from muted palette
                'prep': base_palette[5]  # blue-like color from muted palette
                # 'other': base_palette[4]  # brown-like color from muted palette
            }
            my_subs_order = ['parenth', 'adv_pred', 'conj', 'prep', 'part']
            plot_me = {'uniq': uniq_slangs_df, 'user': user_slangs_df}
        else:  # surprisals for ttypes (source, gold, lit, user) by slang
            # Retrieve the full PuBuGn palette with 10 colors
            same_palette = sns.color_palette("PuBuGn", 8)
            contrastive_palette = sns.color_palette("hls", 8)
            my_palette = {
                'gold': same_palette[1],  # grey
                'lit': same_palette[7],  # teal (dark greenish)
                'src': same_palette[4],  # light blue
                'user': contrastive_palette[0],  # zinnober red-orange
            }
            my_subs_order = ['src', 'gold', 'lit', 'user']

            plot_me = {'uniq': uniq_slangs_df, 'user': user_slangs_df}

        if args.exp == 'category':
            outname = f'srp-user-vs-source-pos_{args.model}.png'
        elif args.exp == 'normed_annotation':
            outname = f'user_srp_techniques_{args.model}.png'
        else:  # slangs
            outname = f'srp_gold-lit-src-usr_{args.model}.png'

        # exp: slangs, category, annotation
        boxplot_srp(datas=plot_me, my_exp=args.exp, outdir=picsto, outf=outname,
                    colours=my_palette, lang_order=LANGUAGE_ORDER,
                    subs_order=my_subs_order, text=None,
                    lose_outliers=False)

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
