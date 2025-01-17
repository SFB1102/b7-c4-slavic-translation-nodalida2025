"""
2 October 2024
after the data was built in final_wide_tables/ import these variables where needed

from configs import SLANG_MAP, PREDICTORS, DESCRIPTORS, LANGUAGE_ORDER, POS_ORDER, ANNO_ORDER, MODEL
"""

SLANG_MAP = {'CS': 'Czech', 'PL': 'Polish', 'BG': 'Bulgarian', 'UK': 'Ukrainian', 'BE': 'Belarusian'}
PREDICTORS = ['surprisal_src', 'surprisal_lit',
              'surprisal_gold',
              'surprisal_src_sent', 'surprisal_lit_sent', 'surprisal_gold_sent',
              'cosine_src-gold', 'cosine_src-lit',
              'pwld_original_lit', 'pwld_gold_lit', 'pwld_gold_original',
              'qe_gold', 'eval_lit', 'qe_lit']
DESCRIPTORS = ['surprisal_user', 'surprisal_user_sent', 'cosine_src-user',
               'eval_user']  # , 'qe_user', 'entropy', 'user_vars_num'

LANGUAGE_ORDER = ['Czech', 'Polish', 'Bulgarian', 'Belarusian', 'Ukrainian']
POS_ORDER = ['parenth', 'adv_pred', 'conj', 'prep', 'part']
ANNO_ORDER = ['correct', 'fluent_literal', 'paraphrase', 'awkward_literal', 'fantasy', 'noise', 'empty']
MODEL = 'ruRoBERTa-large'  # used to create subdirs: 'rugpt3large_based_on_gpt2', 'ruRoBERTa-large', 'xlm-roberta-base'

