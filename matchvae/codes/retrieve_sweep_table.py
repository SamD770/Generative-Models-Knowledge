import wandb
import json
from train import train_args_parser
api = wandb.Api()

args = train_args_parser()

# get entity and project
with open(args.wandb_config) as f:
    wandb_config = json.load(f)

# login
wandb.login(key=wandb_config['key'])

# get sweep id
sweep_id = args.sweep_id
sweep_id = '{}/{}/{}'.format(wandb_config['entity'], wandb_config['project'], sweep_id)

# get results
runs = api.sweep(sweep_id).runs
summary_list = []
config_list = []
name_list = []
id_list = []
for run in runs:
    # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # run.config is the input metrics.  We remove special values that start with _.
    config_list.append({k:v for k,v in run.config.items() if not k.startswith('_')})

    # run.name is the name of the run.
    name_list.append(run.name)

    # run id
    id_list.append(run.id)

import pandas as pd
summary_df = pd.DataFrame.from_records(summary_list)
config_df = pd.DataFrame.from_records(config_list)
name_df = pd.DataFrame({'name': name_list})
id_df = pd.DataFrame({'ID': id_list})
all_df = pd.concat([name_df, id_df, config_df,summary_df], axis=1)

all_df.to_csv("../outputs/sweep_outputs/{}.csv".format(args.table_name))