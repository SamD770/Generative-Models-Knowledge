# Reproducibility code

## Part 1 : Modules

The list of modules in the Python environment used to write this code is available in `environment.yml`. Specifically, it is recommended to use Ubuntu 20.04, Python 3.8, CUDA 11.1 (which is a prerequisite to even import the environment using `environment.yml` ) and the same versions of PyTorch, PyTorch-Lightning, JAX, numpy, pandas and wanbai (see next part) as those from the `environment.yml` file.

Specifically, move to the root folder of the code and run :

`conda env create -f environment.yml`.

If `jaxlib` can't be installed in the process and makes this command crash, then open `environment.yml` and change the line with `jaxlib` to 

`    - jaxlib==0.1.65` 

by removing the `+` component with the CUDA specification, and respecting the indentation. That way, JAX will be run in CPU mode. If you still wish to use CUDA with JAX, run `bash install_jax.sh`. Change  `install_jax.sh` according to the instructions [here](https://github.com/google/jax#installation) if you use another version of CUDA.



## Part 2 : Weights & Biases

The code to generate results for VAE models (MatchVAE and plain VAE) relies on [Weights & Biases (W&B)](https://wandb.ai). It was tested on a free account.

1. Create a free account if you do not have any account on W&B.

2. Create a dedicated project for the reproducibility code. The project will then have a name under the format `<entity>/<project name>` where `<entity>` usually is the W&B username, but can also be the name of a team if the project is created under that team.

3. Find your API key in **Settings > API keys**

4. Fill the JSON file `codes/wandb_config.json` using these indications :

   ```yaml
   {
     "entity": "<The <entity> part in the <entity>/<project name> format from step 2, usually the W&B username from step 1>",
     "project": "<The <project name> part in the <entity>/<project name> format from step 2>",
     "key": "<The API key from step 3>"
   }
   ```

   

## Part 3 : Training VAE models

For this part, it is recommended to have at least 50GB of free space on the drive. If not, it is required to have at least 12GB of free space on the drive and empty the folder `outputs/wandb/` after each of steps 3, 4, 5, 6 just below.

1. If not using a GPU, open `codes/train.py `and replace the line 41 :

   ```python
       parser.add_argument('--device', type=str, default="cuda", metavar='N',  # e.g. "cuda", "cpu", ...
   ```

   with

   ```python
       parser.add_argument('--device', type=str, default="cpu", metavar='N',  # e.g. "cuda", "cpu", ...
   ```



2. Go to `codes/`.

3. To obtain the data with results for MatchVAE on synthetic datasets :
   1. Run `python sweep_config.py --sweep_config sweep_configs/matchvae_synthetic1.yaml`.
   2. Note the 8-character ID of the sweep, or find it on the sweep's page in W&B.
   3. Optionally, run `python sweep_config.py --sweep_id XXXXXXXX` where `XXXXXXXX` is the ID of the sweep in one or several other terminals to speed up the grid search.
   4. Save the results by running `python retrieve_sweep_table.py --sweep_id XXXXXXXX --table_name synthetic1` where `XXXXXXXX` is the ID of the sweep. Alternatively, go to the "Table" section of the W&B page of the sweep, and export the table as `outputs/sweep_outputs/news.csv`. 

4. To obtain the required model data for MatchVAE on News datasets :
   1. Run `python sweep_config.py --sweep_config sweep_configs/matchvae_news.yaml`.
   2. Note the 8-character ID of the sweep, or find it on the sweep's page in W&B.
   3. Optionally, run `python sweep_config.py --sweep_id XXXXXXXX` where `XXXXXXXX` is the ID of the sweep in one or several other terminals to speed up the grid search.
   4. Save the results by running `python retrieve_sweep_table.py --sweep_id XXXXXXXX --table_name news` where `XXXXXXXX` is the ID of the sweep. Alternatively, go to the "Table" section of the W&B page of the sweep, and export the table as `outputs/sweep_outputs/news.csv`. 

5. To obtain the required model data for the plain VAE on synthetic datasets :
   1. Run `python sweep_config.py --sweep_config sweep_configs/plainvae_synthetic1.yaml`. Make sure W&B run data is being saved in `outputs/wandb/`.
   2. Note the 8-character ID of the sweep, or find it on the sweep's page in W&B.
   3. Optionally, run `python sweep_config.py --sweep_id XXXXXXXX` where `XXXXXXXX` is the ID of the sweep in one or several other terminals to speed up the grid search.
   4. Save the run information by running `python retrieve_sweep_table.py --sweep_id XXXXXXXX --table_name plainvae_synthetic1_infos` where `XXXXXXXX` is the ID of the sweep. Alternatively, go to the "Table" section of the W&B page of the sweep, **include the "ID" name in columns** and export the table as `outputs/sweep_outputs/plainvae_synthetic1_infos.csv`. 

6. To obtain the required model data for the plain VAE on News datasets :
   1. Run `python sweep_config.py --sweep_config sweep_configs/plainvae_news.yaml`. Make sure W&B run data is being saved in `outputs/wandb/`.
   2. Note the 8-character ID of the sweep, or find it on the sweep's page in W&B.
   3. Optionally, run `python sweep_config.py --sweep_id XXXXXXXX` where `XXXXXXXX` is the ID of the sweep in one or several other terminals to speed up the grid search.
   4. Save the results by running `python retrieve_sweep_table.py --sweep_id XXXXXXXX --table_name plainvae_news_infos` where `XXXXXXXX` is the ID of the sweep. Alternatively, go to the "Table" section of the W&B page of the sweep, **include the "ID" name in columns** and export the table as `outputs/sweep_outputs/plainvae_news_infos.csv`. 

## Part 4 : Generating outcome regression results

Now we generate ATT errors from outcome regression competitors.

1. Go to `codes/benchmarks/CATENets`.
2. To generate results on synthetic datasets, run `python run_experiments.py --experiment synthetic1`.
3. To generate results on News datasets, run `python run_experiments.py --experiment news`.

## Part 5 : Generating matching results other than MatchVAE

Now we generate ATT errors and imbalances from matching competitors.

1. Go to `codes/benchmarks/matchings`.
2. To generate matching results for the plain VAE on synthetic datasets, run `python plainvae_matchings_synthetic1.py`. 
3. To generate matching results for the plain VAE on News datasets, run `python plainvae_matchings_news.py`.
4. To generate matching results for other methods than plain VAE on synthetic datasets, run `python matchings_synthetic1.py`.  
5. To generate matching results for other methods than plain VAE on News datasets, run `python matchings_news.py`.  

## Part 6 : Visualising results

1. To visualise ATT error and imbalance results on synthetic datasets (Appendix D), open and execute the Jupyter notebook `notebooks/results_synthetic1.ipynb`.
2. To visualise ATT error and imbalance results on News datasets (Appendix D), open and execute the Jupyter notebook `notebooks/results_news.ipynb`.
3. To visualise normalised Mann-Whitney U-statistic plots (Appendix C), open and execute the Jupyter notebook `notebooks/mw_u_statistics.ipynb`.