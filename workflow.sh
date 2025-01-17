# produce final data: add unique IDs, get GPT-4 literal translations, normalise spelling (lowercase, delete spelling errors),
# pre-annotate and get pre-annotation stats by source lang
python3 gpt_literal_pre_anno.py --intab data/input/revised_annotated_curated.tsv
# Figure 1. Bars for translation solutions and line plots for mean PWLD between original and gold MSUs, and between gold and literal variants,
# with PWLD values on left y-axis.
python3 bars_solutions_pwld.py --overlay_distance

python3 comet_evaluation.py --model Unbabel/wmt22-comet-da
python3 comet_evaluation.py --model Unbabel/wmt22-cometkiwi-da --qe

python3 main_surprisal.py --model ruRoBERTa-large --exp slangs

python3 get_similarities.py

python3 calculate_entropy.py --entropy_type user

python3 collect_features.py --entropy_type user --how user
python3 collect_features.py --entropy_type user --how uniq

python3 literality_bars.py --overlay --strategy pwld_top33
python3 literality_bars.py --overlay --strategy gold=lit

python3 violin_plots.py --data uniq

python3 uniq_regressor.py --how SVR --nbest 5 --y_column intelligibility
python3 uniq_regressor.py --how SVR --nbest 5 --y_column entropy

python3 univariate_analysis.py