"""
22 Sept 2024
uses the output of main_surprisal.py

python3 get_similarities.py

"""
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import argparse
import os
import time
from datetime import datetime
from minicons import cwe

from tqdm import tqdm

import torch
import torch.nn.functional as F

import warnings
from unidecode import unidecode
import re
from configs import MODEL

warnings.filterwarnings("ignore")

def get_all_parallel_surprisal_table(uniq=None, user=None, failed_from=None, outf=None):
    # noise-empty from
    failed_df = pd.read_csv(failed_from, sep='\t')
    all_failed_ids = failed_df['ID'].tolist()
    print(len(all_failed_ids))
    failed_ids = [i for i in all_failed_ids if not i in user['ID'].unique()]
    print(len(failed_ids))

    collector = []
    for i in user['ID'].tolist() + failed_ids:
        for ii in uniq['ID'].unique():
            if i.split(':')[0] == ii.split(':')[0]:
                # print(i.split(':')[0])
                # get a one row df
                if i in failed_ids:
                    temp = failed_df[failed_df['ID'] == i][['ID', 'category', 'language', 'normed_annotation']].reset_index(drop=True)
                    user_dict = {'user': np.nan, 'surprisal_user': np.nan, 'user_sent': np.nan,
                                 'surprisal_user_sent': np.nan}
                    temp_user_row = pd.DataFrame(user_dict, index=[0])
                    user_row = pd.concat([temp, temp_user_row], axis=1)
                else:
                    user_row = user[user['ID'] == i].reset_index(drop=True)
                uniq_row = uniq[uniq['ID'] == ii].reset_index(drop=True)
                uniq_row = uniq_row[['src', 'surprisal_src', 'src_sent', 'surprisal_src_sent', 'gold', 'surprisal_gold',
                                     'gold_sent', 'surprisal_gold_sent', 'lit', 'surprisal_lit', 'lit_sent',
                                     'surprisal_lit_sent']]
                enriched_row = pd.concat([uniq_row, user_row], axis=1)

                enriched_row = enriched_row[['ID', 'category', 'language', 'src', 'surprisal_src', 'src_sent',
                                             'surprisal_src_sent', 'gold', 'surprisal_gold', 'gold_sent',
                                             'surprisal_gold_sent', 'lit', 'surprisal_lit', 'lit_sent',
                                             'surprisal_lit_sent', 'normed_annotation', 'user', 'surprisal_user',
                                             'user_sent', 'surprisal_user_sent']]
                # print(enriched_row)

                collector.append(enriched_row)

    wide_df = pd.concat(collector, axis=0)
    if str(len(collector)) in outf:
        wide_df.to_csv(outf, sep='\t', index=False)
    else:
        wide_df.to_csv(outf.replace('.tsv', f'_{len(collector)}.tsv'), sep='\t', index=False)
        print('Some instances are missing? Not 6579?')
        print(len(collector))
        exit()

    return wide_df


def mean_pool_msu_vector(msu_words, sent_words, listed_vectors=None):
    extracted_form = None
    msu_vector = None
    try:
        # Extract indices of all occurrences of the 1st item
        possible_start_indices = [i for i, x in enumerate(sent_words) if x == msu_words[0]]
        if len(possible_start_indices) != 1:
            if len(msu_words) > 1:  # if MSU is multiword! not the case in BG досега and its' user's variant почему
                for start_idx in possible_start_indices:
                    next_idx = start_idx + 1
                    if sent_words[next_idx] == msu_words[1]:
                        end_idx = start_idx + len(msu_words)
                        # stacked_tensors = torch.cat(listed_vectors[start_idx:end_idx], dim=0)
                        stacked_tensors = torch.cat(listed_vectors[start_idx:end_idx], dim=0)
                        # loosing the vector with nan: I get it got proposition 'по' from Russian models! in
                        # подозреваемых до сих пор было восемь человек , из них двое , по моему мнению , исключались по психологическим мотивам
                        stacked_vectors_np = stacked_tensors.numpy()
                        msu_vector = torch.tensor(np.nanmean(stacked_vectors_np, axis=0))
                        # msu_vector = torch.mean(stacked_tensors, dim=0)  # mean_pooled_tensor
                        extracted_form = ' '.join(sent_words[start_idx:end_idx])
                    else:
                        continue
            else:
                # досега is 2nd почему
                # куда исчез, почему – почему загадка. а команда шикина разбрелась кто куда.
                oneword_msu_last_idx = len(sent_words) - 1 - sent_words[::-1].index(msu_words[0])
                msu_vector = listed_vectors[oneword_msu_last_idx]
                extracted_form = sent_words[oneword_msu_last_idx]
        elif len(possible_start_indices) == 0:
            print('First component of the MSU is not in sentence? Whaaaat??')
            # print(sent_words)
            # print(msu_words)
            exit()
        else:
            start_idx = sent_words.index(msu_words[0])
            end_idx = start_idx + len(msu_words)
            stacked_tensors = torch.cat(listed_vectors[start_idx:end_idx], dim=0)
            # loosing the vector with nan among those to sum: I get it got proposition 'по' from Russian models! in
            # подозреваемых до сих пор было восемь человек , из них двое , по моему мнению , исключались по психологическим мотивам
            stacked_vectors_np = stacked_tensors.numpy()
            msu_vector = torch.tensor(np.nanmean(stacked_vectors_np, axis=0))
            # msu_vector = torch.mean(stacked_tensors, dim=0)  # mean_pooled_tensor
            extracted_form = ' '.join(sent_words[start_idx:end_idx])
    except ValueError:
        print(f'I cannot find "{msu_words}" in "{sent_words}".')
        exit()
    except IndexError:
        print('Index error!')
        print(sent_words)
        print(msu_words)

    if ' '.join(msu_words) != extracted_form:
        print('\nYou extracted surprisal for something else!')
        print()
        print(' '.join(msu_words))
        print(extracted_form)
        print(sent_words)
        exit()

    return msu_vector


def define_words(item, mmodel=None):
    # clean up a bit to avoid differences in tokenisation between the phrase and the sent
    # will a universal max clean-up approach work? This is what was necessary for surprisal from rugpt3
    item = item.replace('…', '').replace('.', '').replace('!', '').replace('%', '').replace('-', ' - ')
    item = item.replace('(', '').replace(')', '').replace('«', '').replace('»', '').replace('–', '')
    item = item.replace('―', '').replace('?..', '?').replace('pr.n.l.', 'prnl')

    words = re.findall(r'\d+(?:[.,]\d+)?\.?|\w+(?:\.\w+)*\b(?:\.)?|[^\w\s](?<!\.)', item, re.UNICODE)

    return words


def get_model_cosines(slang_df, my_slang=None, paired_cols=None, mmodel=None, temp_dir=None):
    # adding columns: 'cosine_src-gold', 'cosine_src-lit', 'cosine_src-user'
    this_lang_cos = slang_df.copy()
    err_dict = defaultdict(int)
    my_ttype_vectors = defaultdict()
    for cols in paired_cols:

        slang_ttype_save_as = f'{temp_dir}{my_slang}_{cols[0]}_{mmodel}.pkl'
        slang_ttype_err_save_as = f'{temp_dir}{my_slang}_{cols[0]}_err_{mmodel}.pkl'
        try:
            this_type_slang_msu_vectors = restore(slang_ttype_save_as)
            err_dict = restore(slang_ttype_err_save_as)
            print(f'\n== Vectors from {args.model} for {my_slang.upper()} {cols[0]} are restored from pickle ==')
        except FileNotFoundError:
            print(f'\n== Vectors for {my_slang.upper()} {cols[0]} are being inferred anew ==')

            this_type_slang_msu_vectors = []

            if cols[0] == 'user':
                form_col_name, sent_col_name, anno_col = cols
                annotations = slang_df[anno_col].tolist()
                sents = slang_df[sent_col_name].tolist()
                forms = slang_df[form_col_name].tolist()
                tuple_list = list(zip(sents, forms, annotations))
            else:
                form_col_name, sent_col_name = cols
                sents = slang_df[sent_col_name].tolist()
                forms = slang_df[form_col_name].tolist()
                tuple_list = list(zip(sents, forms))

            for tup in tqdm(tuple_list, position=0, leave=True):
                word_vectors = []
                if len(tup) > 2:
                    if tup[2] not in ['empty', 'noise']:  # user
                        # vectorise words in the sent:
                        try:
                            sent_words = define_words(tup[0], mmodel=mmodel)
                            msu_words = define_words(tup[1], mmodel=mmodel)
                            for w in sent_words:
                                context_word_emb = minicons_vectoriser.extract_representation([(' '.join(sent_words), w)], layer=12)
                                word_vectors.append(context_word_emb)
                        except AttributeError:
                            print(tup)
                            print(tup[2])
                            exit()

                        msu_mean_pooled_vector = mean_pool_msu_vector(msu_words, sent_words, listed_vectors=word_vectors)
                        if msu_mean_pooled_vector is None or torch.isnan(msu_mean_pooled_vector).any():
                            print("Vector is either None or contains NaNs-1.")
                            err_dict[form_col_name] += 1
                        this_type_slang_msu_vectors.append(msu_mean_pooled_vector)
                    else:
                        this_type_slang_msu_vectors.append(None)
                else:  # src, gold, lit
                    # vectorise words in the sent:
                    try:
                        sent_words = define_words(tup[0], mmodel=mmodel)
                        msu_words = define_words(tup[1], mmodel=mmodel)
                        for w in sent_words:
                            try:
                                context_word_emb = minicons_vectoriser.extract_representation([(' '.join(sent_words), w)], layer=12)
                                word_vectors.append(context_word_emb)
                            except AssertionError:
                                print(w)
                                print(' '.join(sent_words))
                                exit()
                    except AttributeError:
                        print(tup)
                        print(tup[2])
                        exit()

                    msu_mean_pooled_vector = mean_pool_msu_vector(msu_words, sent_words, listed_vectors=word_vectors)
                    if msu_mean_pooled_vector is None or torch.isnan(msu_mean_pooled_vector).any():
                        print("Vector is either None or contains NaNs-2.")
                        err_dict[form_col_name] += 1
                    this_type_slang_msu_vectors.append(msu_mean_pooled_vector)

            # pickle to avoid learning vectors on each run with the same input
            save(filename=slang_ttype_save_as, vect=this_type_slang_msu_vectors)
            save(filename=slang_ttype_err_save_as, vect=err_dict)
        my_ttype_vectors[cols[0]] = this_type_slang_msu_vectors
    if len(my_ttype_vectors['src']) == len(my_ttype_vectors['gold']) == len(my_ttype_vectors['lit']) == len(my_ttype_vectors['user']):
        print('I have the same N of vectors for src gold lit phrases! YaY!')

    for pair in ['src-gold', 'src-lit', 'src-user']:
        cos_col_name = f'cosine_{pair}'
        cosine_vals = []
        for emb1, emb2 in zip(my_ttype_vectors[pair.split("-")[0]], my_ttype_vectors[pair.split("-")[1]]):
            if emb1 is not None and emb2 is not None and emb1.numel() > 0 and emb2.numel() > 0:
                emb1 = emb1.unsqueeze(0)  # shape: [1, 768]
                emb2 = emb2.unsqueeze(0)  # shape: [1, 768]
                value = cosine_similarity(emb1, emb2)
            else:
                value = None
            cosine_vals.append(value)

        this_lang_cos.insert(0, cos_col_name, cosine_vals)

    error_df = pd.DataFrame(err_dict, index=[0])

    return this_lang_cos, error_df


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
        if current_word == words[word_idx]:
            aggregated_words.append(current_word)  # Add the full word
            aggregated_values.append(current_value_sum)  # Add the sum of values for the word

            # Reset for the next word
            current_word = ""
            current_value_sum = 0.0
            word_idx += 1  # Move to the next word

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


def cosine_similarity(embedding1, embedding2):
    # Compute cosine similarity
    similarity = F.cosine_similarity(embedding1, embedding2, dim=-1)
    return similarity.item()


def clean_text(text):
    return ''.join([char if char.isprintable() else '' for char in text])


def save(filename=None, vect=None):
    pickle.dump(vect, open(filename, 'wb'), pickle.HIGHEST_PROTOCOL)


def restore(filename=None):
    vect = pickle.load(open(filename, 'rb'))
    return vect


def make_dirs(logsto=None, temp=None, pics=None, res=None, user_srp_to=None, mmodel=None):
    os.makedirs(user_srp_to, exist_ok=True)
    logs_to = f'{logsto}{mmodel}/'
    os.makedirs(logsto, exist_ok=True)
    res_to = f'{res}{mmodel}/'
    os.makedirs(res_to, exist_ok=True)
    temp_to = f'{temp}{mmodel}/'
    os.makedirs(temp_to, exist_ok=True)
    current_datetime = datetime.utcnow()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H:%M')
    script_name = sys.argv[0].split("/")[-1].split(".")[0]
    log_file = f'{logs_to}{formatted_datetime.split("_")[0]}_{script_name}.log'
    sys.stdout = Logger(logfile=log_file)
    print(f"\nRun date, UTC: {datetime.utcnow()}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")

    return temp_to, res_to, user_srp_to


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
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cats_from', help="main df", default='data/normalised_translation_experiment_data.tsv')
    parser.add_argument('--srp', default=f'res/srp/{MODEL}/')
    parser.add_argument('--model', default=MODEL)  # 'ruRoBERTa-large'
    parser.add_argument('--res', default=f'res/emb/')
    parser.add_argument('--temp', default=f'temp/emb/')
    parser.add_argument('--logs', default='logs/emb/')

    args = parser.parse_args()

    start = time.time()
    tempto, resto, user_to = make_dirs(logsto=args.logs, temp=args.temp, res=args.res,
                                       user_srp_to=f'{args.srp}user_view/', mmodel=args.model)

    if args.model == 'xlm-roberta-base':
        minicons_vectoriser = cwe.CWE(args.model)
    else:
        minicons_vectoriser = cwe.CWE(f'ai-forever/{args.model}')

    DIACRITIC_LANGUAGES = ['PL', 'CS']

    # transform the default main_surprisal output and save a temp file to avoid genrating it again for calculating entropy!
    uniq_slangs_df = pd.read_csv(f'{args.srp}wide_srp-{args.model}_src-lit-gold.tsv', sep='\t')
    user_slangs_df = pd.read_csv(f'{args.srp}wide_srp-{args.model}_user.tsv', sep='\t')
    save_user_as = f'{user_to}user_6579.tsv'
    df0 = get_all_parallel_surprisal_table(uniq=uniq_slangs_df, user=user_slangs_df,
                                           failed_from=args.cats_from, outf=save_user_as)
    print(df0.head())
    exit()
    save_as = f'{resto}cosines_{args.model}.tsv'
    errors_save_as = f'{resto}errors_cosines_{args.model}.tsv'

    if os.path.exists(save_as):
        cosines_df = pd.read_csv(save_as, sep='\t')
        srp_err_df = pd.read_csv(errors_save_as, sep='\t')
        print(f'\n== Tables with cos similarities from {args.model} are restored from pickle ==')
    else:
        print(f'\n== Inferring embeddings and calculating cosines from {args.model} model anew ==')

        collector = []

        tot_err_dict = defaultdict()
        err_collector = []
        print(df0.language.unique())

        for slang in df0.language.unique():
            print(slang)
            lang_save_as = save_as.replace('res/', 'temp/').replace('cosines_', f'{slang}_cosines_')
            lang_errors_save_as = errors_save_as.replace('res/', 'temp/').replace('errors_', f'{slang}_errors_')
            if os.path.exists(lang_save_as):
                enriched_slang = restore(lang_save_as)
                err_report = restore(lang_errors_save_as)
                print(f'== DataFrame with cosines for {slang.upper()} is restored from pickle ==')
            else:
                print(f'== Inferring embeddings for {slang.upper()} model anew ==')
                this_slang = df0[df0.language == slang]

                if slang in DIACRITIC_LANGUAGES:
                    for col in ["src", "src_sent"]:
                        this_slang[col] = this_slang[col].apply(unidecode)
                # є problem in єдиної: 'яно', 'ї', 'спор', 'уди', ',', '�', '�', 'дино', 'ї', 'кам', "'",
                if slang == 'UK':
                    for col in ["src", "src_sent"]:
                        # I tried normalize, explicitly handle encoding conversion, clean_text
                        this_slang[col] = this_slang[col].apply(lambda x: x.replace('є', 'е'))
                # user includes noise and empty!
                _cols = [['src', 'src_sent'], ['gold', 'gold_sent'], ['lit', 'lit_sent'], ['user', 'user_sent', 'normed_annotation']]

                # adding columns: 'cosine_src_gold', 'cosine_src_lit', 'cosine_src_user' for the entire data!
                # there will be Nones for noise and empty in cosine_src_user
                enriched_slang, err_report = get_model_cosines(this_slang, my_slang=slang,
                                                               paired_cols=_cols, mmodel=args.model,
                                                               temp_dir=tempto)

                err_report.insert(0, 'slang', slang)

                # pickling output for each language
                save(filename=lang_save_as, vect=enriched_slang)
                save(filename=lang_errors_save_as, vect=err_report)

            collector.append(enriched_slang)
            err_collector.append(err_report)

            print(f'Data with cosines for {slang}:')
            print(enriched_slang.columns.tolist())
            print(enriched_slang.shape)
            print(f'*** {slang}: Cosines btw src and all translations (gold, lit, user) are ready. ***\n')

        cosines_df = pd.concat(collector, axis=0)

        final_msg = "== Tables with srp pickled ==" \
                    f"\n\tShape: {cosines_df.shape}, Location: {save_as}" \
                    f"\n\tColumns: {cosines_df.columns.tolist()}"
        print(final_msg)

        srp_err_df = pd.concat(err_collector, axis=0)
        print(f'\n** {args.model} ** Errors in surprisal inference:')
        print(srp_err_df)

        # save as df
        cosines_df.to_csv(save_as, sep='\t', index=False)
        srp_err_df.to_csv(errors_save_as, sep='\t', index=False)

        vec_tot = time.time()
        print(f"\nTotal calculation time: {round((vec_tot - start) / 60, 2)} min")

    print('\nFinal results table:')

    print(cosines_df.tail())
    print(cosines_df.shape)

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
