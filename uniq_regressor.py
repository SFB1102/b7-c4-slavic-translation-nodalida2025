"""
16 Jan UPD: Figure 4 in the paper
Tables 2 (response variable == intelligibility)
Tables 3 (response variable == entropy)

Top features for Table 5 in the Appendix

13 Jan
treat entropy as a response variable alternative to

python3 uniq_regressor.py

"""

import numpy as np
import os
import sys
import pandas as pd
import argparse
import time
from datetime import datetime
from collections import Counter
from collections import defaultdict

from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.feature_selection import RFE, RFECV

from sklearn.linear_model import HuberRegressor, LinearRegression, SGDRegressor
from sklearn.dummy import DummyRegressor

from sklearn.utils import shuffle

from scipy.stats import pearsonr, spearmanr

from configs import SLANG_MAP, PREDICTORS, MODEL, LANGUAGE_ORDER


# Function to map indices to feature names
def replace_indices_with_names(indices_list, feats_df):
    return [feats_df.columns[idx] for idx in indices_list]


# NB! they return tuples of score:significance, these functions need to return ONE value
def my_pearson_r(trues, predictions):
    return pearsonr(trues, predictions)[0]  # r, p


def my_spearman_r(trues, predictions):
    return spearmanr(trues, predictions)[0]


def my_rmse(trues, predictions):
    return mean_squared_error(trues, predictions, squared=False)


def recursive_elimination(_x, _y, feats=None, my_algo=None, cv=None, scaling=None):
    if my_algo == 'linSVR' or my_algo == 'SVR':  # I cannot use a more complex SVM for feature selection:
        # it has no feature_importance attribute!
        regressor = SVR(kernel="linear", C=1.0)
    # elif my_algo == 'SVR':
    #     regressor = SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0,
    #                     epsilon=0.1, max_iter=-1, shrinking=True)
        # scoring = make_scorer(lambda x, y: pearsonr(x, y)[0], greater_is_better=True)
    else:
        raise ValueError("Only 'SVR' is currently supported for recursive elimination.")

    if feats == -1:  # Use RFECV
        print(f'Running feature selection with {my_algo} in {cv}-fold cross-validation setting')
        selector = RFECV(estimator=regressor, step=1, cv=cv, scoring='r2', n_jobs=-1, min_features_to_select=1)
    else:  # Use RFE with a fixed number of features
        print(f'Selecting top {feats} features with {my_algo}')
        selector = RFE(estimator=regressor, n_features_to_select=feats, step=1)

    # Define pipeline
    if scaling:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Step 1: Scale features
            ('feature_selection', selector)  # Step 2: Perform RFE(CV)
        ])
    else:
        pipeline = Pipeline([
            ('feature_selection', selector)
        ])

    # Fit the pipeline
    pipeline.fit(_x, _y)
    # Access the RFECV step in the pipeline
    rfecv_step = pipeline.named_steps['feature_selection']

    feature_ranking = rfecv_step.ranking_
    # Filter and sort selected features based on their rank
    selected_features_with_ranks = [(idx, rank) for idx, rank in enumerate(feature_ranking) if rank == 1]
    # Sort features by their importance (if you have additional importance metrics)
    # Otherwise, they are naturally ranked since all have rank == 1
    selected_features_sorted = sorted(selected_features_with_ranks, key=lambda x: x[1])

    top_idx = [idx for idx, _ in selected_features_sorted]

    # Get the boolean mask of selected features
    # selected_features_mask = rfecv_step.support_
    # top_idx = [idx for idx, selected in enumerate(selected_features_mask) if selected]

    return top_idx


def get_xy_best(training_set, y0, n_feats=None, selector=None, scaling=True, _algo=None, cv=10):
    if selector not in ['RFE', 'RFECV']:
        raise ValueError("Choose a valid selector: 'RFE' or 'RFECV'.")

    # Feature selection
    if n_feats is not None or selector == 'RFECV':
        print(f"Starting {selector} with {_algo}...")
        top_feat = recursive_elimination(training_set, y0, feats=n_feats if selector == 'RFE' else -1,
                                         my_algo=_algo, cv=cv, scaling=scaling)
        top_feat_named = [training_set.columns.tolist()[idx] for idx in top_feat]

        new_df = training_set.iloc[:, top_feat]
        x0 = new_df.values
        print(f'Data after feature selection: {x0.shape}')
    else:
        print("No feature selection applied.")
        x0 = training_set.values
        top_feat_named = None

    return x0, y0, top_feat_named


def cross_validated_regressor(x, y, _algo=None, n_folds=None, scaling=None):
    if _algo == 'LR':
        if scaling:
            regressor = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=-1)  # normalize=False,
        else:
            regressor = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=-1)
    elif _algo == 'SVR':
        # library defaults
        regressor = SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0,
                        epsilon=0.1, max_iter=-1, shrinking=True)
    elif _algo == 'linSVR':
        regressor = SVR(kernel="linear", C=1.0)

    elif _algo == 'DUMMY':
        regressor = DummyRegressor(strategy='mean')
    else:
        regressor = None
    # Create a pipeline if scaling is enabled
    if scaling:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', regressor)
        ])
    else:
        pipeline = Pipeline([
            ('regressor', regressor)
        ])

    print(f'Features/dimensions: {x.shape[1]}')

    if algo == 'DUMMY':
        scorers = {
            'mae': make_scorer(mean_absolute_error, greater_is_better=False),
            'rmse': make_scorer(my_rmse, greater_is_better=True)
        }
    else:
        scorers = {'r2': 'r2',  # Built-in R² scorer
                   'pearson': make_scorer(my_pearson_r, greater_is_better=True),
                   'spearman': make_scorer(my_spearman_r, greater_is_better=True),
                   # They are negatively-oriented scores, which means lower values are better.
                   # # avoid the sign-flip of the outcome of the score_func
                   'mae': make_scorer(mean_absolute_error, greater_is_better=False),
                   'rmse': make_scorer(my_rmse, greater_is_better=True)}

    cv_scores = cross_validate(pipeline, x, y, cv=n_folds, scoring=scorers, n_jobs=-1)  # R_2 is used by default for SVR
    # R_2 reflects the variance explained by the entire model (all predictors together).
    # MAE measures the average absolute difference between the predicted and actual values. Report means of the scores?
    # Pearson evaluates the trend alignment between predictions and true values, not prediction accuracy, reflects LINEAR relation

    if algo == 'DUMMY':
        pearsons = None
        spearmans = None
        r2s = None
    else:
        pearsons = cv_scores['test_pearson']
        spearmans = cv_scores['test_spearman']
        r2s = cv_scores['test_r2']

    maes = cv_scores['test_mae']
    rmses = cv_scores['test_rmse']

    # print('Predicting for all (or nbest) features with default model in the cv setting')
    # preds = cross_val_predict(regressor, x, y, cv=n_folds, n_jobs=-1)

    return r2s, pearsons, spearmans, maes, rmses


def get_preselect_vals(training_set, category='ttype', featnames=None, scaling=None):
    y0 = training_set.loc[:, category].values
    fns = training_set['iid'].tolist()
    # drop remaining meta
    training_set = training_set.drop(['iid', 'ttype', 'lang'], axis=1)
    print(f'Number of input features: {training_set.shape[1]}')

    if scaling:
        print(f'===StandardScaler() ===')
        sc = StandardScaler()
        training_set[featnames] = sc.fit_transform(training_set[featnames])

    x0 = training_set[featnames].values
    new_df = training_set[featnames]
    featurelist = featnames

    return x0, y0, featurelist, fns, new_df


def get_single_vals(training_set, category='ttype', my_single='mean_sent_wc', scaling=None):
    y0 = training_set.loc[:, category].values
    fns = training_set['iid'].tolist()
    # drop remaining meta
    training_set = training_set.drop(['iid', 'ttype', 'lang'], axis=1)
    print(f'Number of input features: {training_set.shape[1]}')

    if scaling:
        print(f'===StandardScaler() ===')
        # centering and scaling for each feature
        # transform your data such that its distribution will have a mean value 0 and standard deviation of 1
        # to meet the assumption that all features are centered around 0 and have variance in the same order
        # each value will have the sample mean subtracted, and then divided by the StD of the whole dataset
        sc = StandardScaler()
        # Reshape your data either using array.reshape(-1, 1) if your data has a single feature
        # training_set = training_set[my_single]
        # print(training_set.head())
        #
        # exit()
        training_set[my_single] = sc.fit_transform(training_set[my_single].values.reshape(-1, 1))

    x0 = training_set[[my_single]].values
    new_df = training_set[[my_single]]
    featurelist = [my_single]

    return x0, y0, featurelist, fns, new_df


def list_to_newline_sep(lst):
    return '\n'.join(lst)


# type hinting def add_numbers(x: int = None, y: int = None) -> int:
def check_for_nans(df0: pd.DataFrame = None) -> None:
    # Check for NaN values in each column, drop 96 rows with NaNs in mdd (one-word segments)
    nan_counts = df0.isna().sum()
    nan_counts_filtered = nan_counts[nan_counts != 0]
    total_nan_count = df0.isna().sum().sum()
    if total_nan_count:
        print(f'Total NaNs in the df: {total_nan_count}')
        # Print the number of NaN values for each column
        print("Columns with NaN values:")
        print(nan_counts_filtered)

        # Select rows where either Column1 or Column2 has None
        # filtered_rows = df0[df0['mhd'].isna() | df0['mdd'].isna()]
        # print(filtered_rows[['doc_id', 'seg_num', 'mdd', 'raw_tok']])

        df0 = df0.dropna(subset=['raw'])  # I could not sort out NaNs for 96 one-word segments like Warum ? and Nein !
        print(df0.shape)
        print('Your data table contains NaNs. Think what to do with them. Exiting ...')
        exit()
    else:
        print(f'No NaNs detected across all {df0.shape[1]} columns of the dataframe')


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
    return res_to, pics_to


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tables', help="IDs, labels, SL, SL_TL feats", default='data/final_wide_tables/uniq/')
    parser.add_argument('--entropy_type', choices=['user', 'user_vars', 'normed_annotation'],
                        help="annotation or user variants?", default='user')
    parser.add_argument('--model', choices=['rugpt3large_based_on_gpt2', 'ruRoBERTa-large', 'xlm-roberta-base'],
                        default=MODEL)
    parser.add_argument('--y_column', choices=['entropy', 'intelligibility'], default='entropy')
    parser.add_argument('--how', choices=['SVR', 'LR', 'linSVR'], default='SVR')
    parser.add_argument('--nbest', type=int, default=0, help="Features to select. -1 for the optimal number with RFECV")
    parser.add_argument('--nbest_by', default='RFE', choices=['RFE', 'RFECV'],
                        help="ablation-based selection, inc with experimental k")
    parser.add_argument('--cv', type=int, default=10, help="Number of folds")
    parser.add_argument('--scale', type=int, default=1, choices=[1, 0], help="Do you want to use StandardScaler?")
    parser.add_argument('--rand', type=int, default=42)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--res', default='res/regress/')
    parser.add_argument('--logs', default='logs/regress/')
    parser.add_argument('--pics', default='pics/regress/')

    args = parser.parse_args()
    start = time.time()

    resto, picsto = make_dirs(logsto=args.logs, res=args.res, pics=args.pics, mmodel=args.model)

    meta = ['item_ID', 'language', 'src', 'category', args.y_column]

    my_cols = meta + PREDICTORS

    _df = pd.read_csv(f"{args.tables}scores_ratios_responses_{args.model}.tsv", usecols=my_cols, sep='\t')
    print(_df.head())
    print(_df.columns.tolist())

    print(f'Features ({len(PREDICTORS)}): {PREDICTORS}')

    run_setup = f'{args.how}_{args.nbest_by}{args.nbest}_{args.entropy_type}'

    feat2ind = {feature: idx for idx, feature in enumerate(PREDICTORS)}

    all_slangs_res_collector = []

    for slang in _df.language.unique():
        print(slang)
        df = _df[_df.language == slang]
        df = df.dropna()
        feats_df = df.drop(meta + [args.y_column], axis=1)
        _Y = df[args.y_column].values
        _X = feats_df.values
        _fns = df['item_ID'].values

        selected_names = None
        selected_ind = None
        # I keep feature selection separate, despite potential data leakage,
        # otherwise I have to deal with separate feature sets for each fold, the study is focused on exploring features
        if args.nbest:
            _X, _Y, selected_names = get_xy_best(feats_df, y0=_Y, n_feats=args.nbest,
                                                 selector=args.nbest_by,
                                                 scaling=args.scale, _algo=args.how, cv=args.cv)
            selected_ind = [feat2ind[i] for i in selected_names]

        # shuffle, i.e randomise the order of labels
        X_shuffled, y_shuffled, fns_shuffled = shuffle(_X, _Y, _fns, random_state=args.rand)
        splitter = args.cv

        this_lang_algos_collector = []

        for algo in [args.how]:  # , 'DUMMY'
            r2, pears, spears, maes, rmses = cross_validated_regressor(X_shuffled, y_shuffled, _algo=algo,
                                                                       n_folds=splitter,
                                                                       scaling=args.scale
                                                                       )
            if algo == 'DUMMY':
                av_pearson = None
                av_spearman = None
            else:
                av_pearson = f'{(sum(pears) / len(pears)):.2f}$\pm${np.std(np.array(pears)):.2f}'
                av_spearman = f'{(sum(spears) / len(spears)):.2f}$\pm${np.std(np.array(spears)):.2f}'
            # convert negative MAE scores back to positive
            av_r2 = f'{(sum(r2) / len(r2)):.2f}$\pm${np.std(np.array(r2)):.2f}'
            av_mae = f'{(-sum(maes) / len(maes)):.2f}$\pm${np.std(np.array(maes)):.2f}'
            av_rmses = f'{(sum(rmses) / len(rmses)):.2f}$\pm${np.std(np.array(rmses)):.2f}'

            support = f'{X_shuffled.shape[0]}items:{X_shuffled.shape[1]}feats'

            mean_y = f'{np.mean(y_shuffled):.2f}$\pm${np.std(np.array(y_shuffled)):.2f}'

            res_tuple = [(algo, av_pearson, av_r2, av_mae, mean_y, av_rmses, support, selected_names, selected_ind)]

            res_df = pd.DataFrame(res_tuple, columns=['algo', 'Pearson', '$R^2$', 'MAE', 'mean_y', 'RMSE',
                                                      "support", 'nbest_names', f'nbest{args.nbest}_ID'])
            this_lang_algos_collector.append(res_df)

        this_lang_res = pd.concat(this_lang_algos_collector, axis=0)
        this_lang_res.insert(0, 'language', slang)
        # print(f'Regression on `{args.y_column}` scores achieves (mean for {args.cv} folds):')
        # print(f'Feature selection setup: {run_setup}')
        # print(this_lang_res)
        # input('Press enter to do the next language')
        all_slangs_res_collector.append(this_lang_res)

    all_res = pd.concat(all_slangs_res_collector, axis=0)
    all_res['language'] = all_res['language'].replace(SLANG_MAP)

    pretty_all_res = all_res.copy()
    pretty_all_res['language_order'] = pd.Categorical(all_res['language'], categories=LANGUAGE_ORDER, ordered=True)
    pretty_all_res_sorted = pretty_all_res.sort_values('language_order').drop('language_order', axis=1)
    print(f'Regression on `{args.y_column}` scores achieves (mean for {args.cv} folds):')
    print(f'N Predictors: {len(PREDICTORS)}')
    print(f'Feature selection setup: {run_setup}')

    pretty_all_res_sorted = pretty_all_res_sorted[['language', 'Pearson', 'MAE', 'mean_y',
                                                   'support', 'nbest_names', f'nbest{args.nbest}_ID']]

    print(pretty_all_res_sorted)

    slangs_outf = f'{resto}pretty_results_{args.y_column}_{run_setup}.tsv'
    pretty_all_res_sorted.to_csv(slangs_outf, sep='\t', index=False)

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
