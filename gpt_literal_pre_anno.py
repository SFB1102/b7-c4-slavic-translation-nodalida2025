"""
14 Jan 2025 update
-- Add unique IDs, get GPT-4 literal translations and normalise spelling (lowercase, delete spelling errors)
-- pre-annotate and get pre-annotation stats by source lang
The actual process was iterative, which might result in some minor discrepancies with the actual data.

26 July 2024

Annotation guidelines with examples: https://docs.google.com/document/d/16msjVu7TKsnWkecy1vgu660a5fGkY9VgS2yZyGWNn3k/edit
Manually curated data: https://docs.google.com/spreadsheets/d/1i6oAhs2M_Bn-1AN4-fjdokfk9Ey1HXRFOCZI4Zpg-FU/edit?gid=146101618#gid=146101618

python3 gpt_literal_pre_anno.py --intab data/input/revised_annotated_curated.tsv
"""

import re
import sys
from collections import Counter, defaultdict
import pickle
import numpy as np
import pandas as pd
import argparse
import os
import time
from datetime import datetime
import openai
import seaborn as sns
import matplotlib.pyplot as plt

# Set your OpenAI API key
openai.api_key = ""
LANG_MAP = {'CS': 'Czech', 'PL': 'Polish', 'BE': 'Belarusian', 'UK': 'Ukrainian', 'BG': 'Bulgarien'}


def first_20_characters_coincide(string_list):
    # Extract the first 20 characters of each string
    first_20_chars = [s[:20] for s in string_list]

    # Check if all the extracted substrings are the same
    return all(chars == first_20_chars[0] for chars in first_20_chars)


def generate_item_instance_response_ids(df):
    collector = []
    for l in df.language.unique():
        this_lang = df[df.language == l]
        # phrase ids
        for idx, ph in enumerate(this_lang.phrase.unique()):
            idx += 1
            itm_idx = str(idx).zfill(2)
            this_ph_df = this_lang[this_lang.phrase == ph]
            # sentence ids
            if len(this_ph_df.sentence.unique()) > 1:

                # Sanity check
                result = first_20_characters_coincide(list(this_ph_df.sentence.unique()))
                if result:
                    print(this_ph_df.sentence.unique())
                    exit()
                else:
                    continue
            for sentence_idx, sentence in enumerate(this_ph_df.sentence.unique()):
                sentence_idx += 1
                # string_sentence_idx = str(sentence_idx).zfill(2)
                # Generate a sequence of integers as IDs
                _response_this_ph_df = this_ph_df[this_ph_df.sentence == sentence]
                response_this_ph_df = _response_this_ph_df.copy()
                response_this_ph_df['ID'] = range(1, len(_response_this_ph_df) + 1)

                # Convert the integers to strings and pad with leading zeros (e.g., to a length of 5 characters)
                response_this_ph_df['ID'] = response_this_ph_df['ID'].astype(str).str.zfill(3)
                # {string_sentence_idx}: -- is excessive they have 1 sentence for each phrase
                response_this_ph_df['ID'] = response_this_ph_df['ID'].apply(lambda x: f"{l}_{itm_idx}:{x}")
                # temp_df = response_this_ph_df[['ID']]
                # print(temp_df.head(20))
                # exit()
                collector.append(response_this_ph_df)
    ided_df = pd.concat(collector, axis=0)
    # print(ided_df.columns.tolist())
    # # put ID at 1st column
    ided_df = ided_df[['ID', 'language', 'sentence', 'translated_sentence', 'category', 'phrase', 'gold_translation',
                       'literal_translation', 'literal_translation_gpt', 'actual_translation', 'annotation',
                       'participant_id', 'participant_gender', 'participant_age', 'participant_l1', 'participant_l2',
                       'nllb_translated']]

    return ided_df


def normalize_items_in_sents(row, itm_col=None, sent_col=None):
    normed_phrase = None
    phrase = row[itm_col]
    context_sent = row[sent_col]
    if phrase in context_sent:
        normed_phrase = phrase
    else:
        if phrase.capitalize() in context_sent:
            normed_phrase = phrase.capitalize()
        else:
            # lowercasing is done for entire sentence
            # всё же vs все же
            if len(phrase.split()) == 2:  # proto že <-- také proto, že bych
                normed_phrase = f"{phrase.split()[0]}, {phrase.split()[1]}"
                if normed_phrase not in context_sent:
                    if 'ё' in phrase:
                        print('ёёёё')
                        # normed_phrase = phrase.replace('ё', 'е')
                        print(f'{normed_phrase} -- {context_sent}')
                        exit()
                    else:
                        print('someting else?')
                        print(f'{normed_phrase} -- {context_sent}')
                        exit()
                # assert normed_phrase in context_sent, f'Still not matching:\n\t{normed_phrase} -- {context_sent}'
                # context_sent = context_sent.replace(punctuated_phrase, phrase)
            elif len(phrase.split()) == 3:
                # z tego że -- w czasie, wynikła z tego, że wszyscy najpierw
                # после того(,) как
                normed_phrase = f"{phrase.split()[0]} {phrase.split()[1]}, {phrase.split()[2]}"
                if normed_phrase not in context_sent:
                    print(f'Still not matching (3-element MSU):\n{phrase}\n{context_sent}')
                    normed_phrase = input("Please provide normalized input: ")
                assert normed_phrase in context_sent, f'What else? (3-element MSU):\n{phrase}\n{context_sent}'
            else:
                print(phrase)
                print(context_sent)
                exit()
    if not normed_phrase:
        print(phrase)
        print(context_sent)
        exit()
    return normed_phrase


def normalize_user(row):
    if row['annotation'] in ['empty', 'noise']:
        normed_user = None
    else:
        user = row['actual_translation']
        # Check if the actual_translation is NaN
        if pd.isna(user):
            print('This should not happen!')
            return None

        if contains_cyrillic(user):
            normed_user = user.lower()
        else:
            print('no cyrillic?')
            print(user)
            normed_user = input("Please provide normalized input: ")

        if len(user) > 25:
            print('Length')
            print(row['sentence'])
            print(row['gold_translation'])
            print(row['translated_sentence'])
            print(row['annotation'])
            print(user)
            normed_user = input("Please provide normalized input: ")

    return normed_user


def pre_annotate(row):
    # paraphrase (user != gold != gpt4_literal + anno:correct or paraphrase) creative effort, sign of good understanding
    # correct (user==gold != gpt4_literal) some effort, coincides with the reference translation where literal is perceived as lacking fluency, standard solution
    # fluent_literal (user==gold==gpt4_literal)
    # awkward_literal (user != gold==gpt4_literal and user != gold != gpt4_literal + anno:literal)
    # fantasy
    # noise
    # empty
    current_anno = row['annotation']
    user = row['normed_user']
    gold = row['normed_gold']
    lit = row['gpt4_literal']

    if user == gold:
        if user == lit:
            # user==gold==gpt4_literal
            new_anno = 'fluent_literal'
        else:
            # user==gold != gpt4_literal
            new_anno = 'correct'
    else:
        if user == lit:
            # user != gold == gpt4_literal
            new_anno = 'awkward_literal'
        else:
            # user != gold != gpt4_literal -- depends on annotation!
            if current_anno == 'correct' or current_anno == 'paraphrase':
                new_anno = 'paraphrase'
            elif current_anno == 'literal':
                new_anno = 'awkward_literal'
            else:  # fantasy, empty, noise
                new_anno = current_anno

    return new_anno


def contains_cyrillic(text):
    # Regular expression to detect Cyrillic characters
    return bool(re.search(r'[А-Яа-я]', text))


def calculate_valid_user_variant_entropy(row):
    sample_list = row['normed_user']
    # Count the frequency of each unique solution
    frequency_counts = Counter(sample_list)

    # Calculate the probabilities
    total_count = len(sample_list)
    probabilities = [count / total_count for count in frequency_counts.values()]

    # Calculate the entropy
    entropy = -sum(p * np.log2(p) for p in probabilities)

    return entropy


def format_prompt(item=None, lang=None, gold=None):
    if gold:
        prompt = f'Return a literal word-for-word translation for a phrase in one of the Slavic source languages into Russian.' \
                 f'Take into account an adequate functional translation, which appears after Reference:. ' \
                 f'Your variant should be distinct from the reference, if possible.' \
                 f'\nFor example, for the query: "Czech: krátce řečeno Reference: одним словом", your response should be: кратче говоря.' \
                 f'\nDo not return the query.' \
                 f'\n{LANG_MAP[lang]}: {item}' \
                 f'\nReference: {gold}'
    else:
        prompt = f'Return a literal word-for-word translation for a phrase in one of the Slavic source languages into Russian.' \
                 f'\nFor example, for the query: "Czech: krátce řečeno", your response should be: кратче говоря.' \
                 f'\nDo not return the query.' \
                 f'\n{LANG_MAP[lang]}: {item}'
    return prompt


def generate_output(prompt, model_name, temperature=None):
    mini_map = {'gpt4': 'gpt-4', 'gpt3.5': 'gpt-3.5-turbo'}
    try:
        response = openai.ChatCompletion.create(
            model=mini_map[model_name],  # Specify your model here
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates text."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=10,  # Adjust the max tokens as needed
        )

        output = response.choices[0].message["content"].strip()
    except Exception as e:
        # Handle other types of exceptions
        print(f"An unexpected error occurred: {str(e)}")
        output = None
    return output


def get_gpt_literal(data, model=None, lang=None, phrase_col=None, gold_col=None):
    gpt_lit = defaultdict()
    uniq_df = data.drop_duplicates(subset=phrase_col, keep='first')

    for _, row in uniq_df.iterrows():
        ph = row[phrase_col]
        ref = row[gold_col]
        this_item_prompt = format_prompt(item=ph, lang=lang, gold=None)
        # print(this_item_prompt)
        gpt_literal = generate_output(this_item_prompt, model, temperature=0.7)
        gpt_literal = gpt_literal.replace('Russian: ', '')
        gpt_literal = gpt_literal.replace(' (Russian)', '')
        gpt_literal = gpt_literal.strip()
        print(ph)
        print(ref)
        print(gpt_literal)
        print()
        gpt_lit[ph] = gpt_literal

    data[f'{model}_literal'] = data[phrase_col].map(lambda x: gpt_lit.get(x, None))

    return data


def track_changes(_df=None, base_col=None, my_slang=None, tracker=None):
    # Compare the two columns
    if base_col == 'gold_translation':
        part_base_col = 'gold'
    elif base_col == 'actual_translation':
        part_base_col = 'user'
    else:  # phrase, annotation
        part_base_col = base_col

    diffs = _df[base_col] != _df[f'normed_{part_base_col}']
    num_diffs = diffs.sum()
    tracker['lang'].append(my_slang)
    tracker['data normed'].append(part_base_col)
    tracker['changes'].append(f'{num_diffs} ({(num_diffs / len(this_lang) * 100):.2f}%)')


def save(filename=None, vect=None):
    pickle.dump(vect, open(filename, 'wb'), pickle.HIGHEST_PROTOCOL)


def restore(filename=None):
    vect = pickle.load(open(filename, 'rb'))
    return vect


def update_responses_stats_for_this_lang(data=None, lang=None, my_dict=None):
    my_dict[lang].append(data.shape[0])

    valid_this_lang = data.loc[~data['normed_annotation'].isin(['empty', 'noise'])]
    my_dict[lang].append(len(set(valid_this_lang['normed_user'].tolist())))

    noise_this_lang = data.loc[data['normed_annotation'].isin(['empty', 'noise'])]
    my_dict[lang].append(noise_this_lang.shape[0])

    # Calculate entropy of valid translation solution by item and mean
    entropy_df = data[['normed_phrase', 'normed_user']]
    grouped_entropy_df = entropy_df.groupby('normed_phrase', as_index=False).agg(
        {'normed_user': lambda x: x.tolist()})
    # grouped_entropy_df['user_vars_entropy'] = grouped_entropy_df.apply(calculate_valid_user_variant_entropy, axis=1)
    grouped_entropy_df['user_vars_num'] = grouped_entropy_df['normed_user'].apply(lambda x: len(x))
    grouped_entropy_df.insert(0, 'language', lang)
    print(grouped_entropy_df.head())

    # Write the mean translation solution entropy across all 57-60 items for this lang
    # mean_user_solution_entropy = grouped_entropy_df['user_vars_entropy'].mean()
    # my_dict[lang].append(mean_user_solution_entropy)

    return my_dict


def get_anno_stats(data=None, anno_col=None, lang=None):
    # Count values in normed_annotation column
    annotation_counts = data[anno_col].value_counts().reset_index()
    annotation_counts.columns = [anno_col, 'absolute_frequency']

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


def plot_overall_responses(plot_df=None, outf=None, bar_type=None, show=None):
    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.set_context('paper')
    # Set the default font properties
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 12
    # fig, ax = plt.subplots()
    if bar_type == "grouped":
        width = 0.3  # Width of the bars
        x = np.arange(len(plot_df['language']))  # Label locations

        # Plot the bars
        bar1 = ax.bar(x - width, plot_df['total_responses'], width, label='Total Responses')
        bar2 = ax.bar(x, plot_df['unique_responses'], width, label='Unique Valid Responses')
        bar3 = ax.bar(x + width, plot_df['noisy_responses'], width, label='Noisy Responses')

        # Add labels, title, and legend
        ax.set_xlabel('')
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('')  # 'Total, unique valid, and noisy responses by source language'
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df['language'])
        ax.legend(title='', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        plt.tight_layout()

        # plt.tight_layout(rect=[0, 0, 0.8, 1])

        # Add value labels on top of the bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        add_labels(bar1)
        add_labels(bar2)
        add_labels(bar3)

    else:
        # Plot stacked bar chart
        plot_df.set_index('language').plot(kind='bar', stacked=True)
        plt.legend(['Total Responses', 'Unique Valid Responses', 'Noisy Responses'])
        plt.xlabel('')
        plt.ylabel('Count')

    plt.savefig(outf.replace('.png', f'_{bar_type}'))
    if show:
        plt.show()


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


def make_dirs(logsto=None, make_them=None):
    for i in make_them:
        os.makedirs(i, exist_ok=True)
    logs_to = f"{logsto}"
    os.makedirs(logs_to, exist_ok=True)
    current_datetime = datetime.utcnow()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H:%M')
    script_name = sys.argv[0].split("/")[-1].split(".")[0]
    log_file = f'{logsto}{formatted_datetime.split("_")[0]}_{script_name}.log'
    sys.stdout = Logger(logfile=log_file)
    print(f"\nRun date, UTC: {datetime.utcnow()}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='pre-annotate and translate with NLLB from Slavic langs into Russian')
    parser.add_argument('--intab', help="", default='data/input/revised_annotated_curated.tsv')
    parser.add_argument('--pkl', default="res/gpt4_literal_pkl/")
    parser.add_argument('--model', choices=['gpt3.5', 'gpt4'], default="gpt4")
    parser.add_argument('--pics', default='pics/')
    parser.add_argument('--stats', default='res/anno/')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()

    start = time.time()

    make_dirs(logsto=args.logs, make_them=[args.pkl, args.stats, args.pics, f'{args.pics}tables/'])

    # ['ID', 'language', 'sentence', 'translated_sentence', 'category', 'phrase', 'gold_translation',
    # 'literal_translation', 'literal_translation_gpt', 'actual_translation', 'annotation', 'participant_id',
    # 'participant_gender', 'participant_age', 'participant_l1', 'participant_l2', 'nllb_translated']
    # Skipping line 1407: expected 17 fields, saw 30
    df = pd.read_csv(args.intab, sep='\t', error_bad_lines=False, warn_bad_lines=True)
    print(df.columns.tolist())
    # move everything to lowercase because tokenisation does it anyway, translated sentence is all-lower already
    # chyba że -- chyba ŻE książę niedomaga.
    df['sentence'] = df['sentence'].str.lower()
    df['gold_translation'] = df['gold_translation'].str.lower()  # ради Бога
    df['actual_translation'] = df['actual_translation'].str.lower()  # А то и
    df['sentence'] = df['sentence'].apply(lambda x: x.replace('4, 7%.', '4,7%.'))

    # Apply .strip() to each string element in the DataFrame
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    df = df.drop('ID', axis=1)  # drop non-uniq IDs
    # add unique IDs by SL e.g. CS_01:02
    df = generate_item_instance_response_ids(df)

    slang_responses_dict = defaultdict(list)

    upd_data_collector = []
    slang_anno_stats_collector = []
    changes_tracker = defaultdict(list)
    for slang in df.language.unique():
        _this_lang = df[df.language == slang]
        this_lang = _this_lang.copy()
        for to_update, phrase_col, sent_col in zip(['normed_phrase', 'normed_gold'],
                                                   ['phrase', 'gold_translation'],
                                                   ['sentence', 'translated_sentence']):
            # normalise items and collect slang stats
            this_lang[to_update] = _this_lang.apply(normalize_items_in_sents, axis=1, itm_col=phrase_col,
                                                    sent_col=sent_col)
            # how much change is introduced to the dataset?
            track_changes(_df=this_lang, base_col=phrase_col, my_slang=slang, tracker=changes_tracker)

        this_lang['normed_user'] = _this_lang.apply(normalize_user, axis=1)
        track_changes(_df=this_lang, base_col='actual_translation', my_slang=slang, tracker=changes_tracker)

        this_lang = this_lang.set_index('ID')
        print(this_lang.head(3))

        save_as = f'{args.pkl}{slang}_{args.model}_upd.pkl'
        if os.path.exists(save_as):
            upd_this_lang = restore(save_as)
            print(upd_this_lang.head(3))
            print(f'\n== {slang} table with literal translations from {args.model} are restored from pickle ==\n')
        else:
            print(f'\n== Prompting for items in {slang} anew ==')
            upd_this_lang = get_gpt_literal(this_lang, model=args.model, lang=slang,
                                            phrase_col='normed_phrase', gold_col='normed_gold')
            # combine normed gold, lit, user and normed anno
            print(upd_this_lang.head())
            print(upd_this_lang.columns.tolist())

            # pickle to avoid learning vectors on each run with the same input
            save(filename=save_as, vect=upd_this_lang)

        # modify the content of annotation as follows:
        upd_this_lang['normed_annotation'] = upd_this_lang.apply(pre_annotate, axis=1)
        track_changes(_df=upd_this_lang, base_col='annotation', my_slang=slang, tracker=changes_tracker)

        # collect tables for languages
        upd_data_collector.append(upd_this_lang)

        # now lets collect some ==USER== parameters into a dict
        slang_responses_dict = update_responses_stats_for_this_lang(data=upd_this_lang, lang=slang,
                                                                    my_dict=slang_responses_dict)
        # Now lets collect stats for ==ANNOTATION== categories into a df
        this_anno_stats_df = get_anno_stats(data=upd_this_lang, anno_col='normed_annotation', lang=slang)
        # this is a historical table which calculates the how many rows were preannotated given the PREVIOUS literal!
        # this_anno_detailed_stats = get_fancy_preanno_stats_table(data=upd_this_lang, anno_col='annotation', lang=slang)
        slang_anno_stats_collector.append(this_anno_stats_df)

    # consolidate the updated data table
    final_data_df = pd.concat(upd_data_collector, axis=0)
    final_data_df = final_data_df.reset_index()
    print(final_data_df.head())
    print(final_data_df.columns.tolist())
    final_data_df.to_csv('data/normalised_translation_experiment_data.tsv', sep='\t', index=False)

    # Monitor changes:
    changes_df = pd.DataFrame(changes_tracker)

    # aggregate and plot RESPONSES stats
    columns = ['total_responses', 'unique_responses', 'noisy_responses']  # , 'mean_phrase_translation_entropy'
    resp2plot = pd.DataFrame.from_dict(slang_responses_dict, orient='index', columns=columns)

    # Reset the index to get language for dict keys as a column
    resp2plot.reset_index(inplace=True)
    resp2plot.rename(columns={'index': 'language'}, inplace=True)
    # Set the order of languages
    language_order = ['CS', 'PL', 'BG', 'UK', 'BE']
    resp2plot['language'] = pd.Categorical(resp2plot['language'], categories=language_order, ordered=True)
    resp2plot = resp2plot.sort_values('language')

    outname = f'{args.pics}crude_overview_responses.png'
    plot_overall_responses(plot_df=resp2plot, outf=outname, bar_type='grouped', show=args.verbose)
    resp2plot.to_csv(f'{args.pics}tables/crude_overview_responses.tsv', sep='\t', index=False)

    # aggregate and plot ANNOTATION stats
    anno2plot = pd.concat(slang_anno_stats_collector, axis=0)

    # Group by 'annotation' and calculate the total absolute frequency
    grouped_df = anno2plot.groupby('normed_annotation').agg({'absolute_frequency': 'sum'}).reset_index()
    total = grouped_df.loc[grouped_df['normed_annotation'] == 'Total', 'absolute_frequency'].values[0]

    print(f"\nTotal number of participants responses: {total}\n")

    # Calculate the percentage for each annotation
    grouped_df['percentage'] = grouped_df['absolute_frequency'] / total * 100
    grouped_df['percentage'] = grouped_df['percentage'].map(lambda x: float(f"{x:.2f}"))
    grouped_df.insert(0, 'language', 'all')
    print(grouped_df)

    all_stats_with_tot = pd.concat([anno2plot, grouped_df], axis=0)
    # see a separate script which plots this data:
    # python3 bars_translation_solutions.py --intab res/anno/overview_annotation_stats.tsv
    stats_to = f'{args.stats}overview_annotation_stats.tsv'
    all_stats_with_tot.to_csv(stats_to, sep='\t', index=False)

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
