"""
Author: Alicia Curth
File to run experiments from shell
"""
import argparse

from catenets.experiments.simulations_AISTATS import main_AISTATS
from catenets.experiments.ihdp_experiments import do_ihdp_experiments
from catenets.experiments.news_experiments import do_news_experiments
from catenets.experiments.synthetic1_experiments import do_synthetic1_experiments


def init_arg():
    # arg parser if script is run from shell
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default='news', type=str)
    parser.add_argument("--setting", default=1, type=int)
    parser.add_argument("--models", default=None, type=str)
    parser.add_argument("--n_repeats", default=50, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = init_arg()

    if args.experiment == 'news':
        do_news_experiments(models=args.models, file_name=args.experiment, n_exp=args.n_repeats)
    elif args.experiment == 'synthetic1':
        do_synthetic1_experiments(models=args.models, file_name=args.experiment, n_exp=args.n_repeats)
