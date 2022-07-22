
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

# PEHE results for in-sample and out_sample
results = pandas.read_csv('CATENets/results/ihdp/results_10rep.csv')
results_old = results

n_experiments = 10  # CHOOSE!
results['experiment'] = list(range(n_experiments))

# do wrangling to required format
results = results.melt(id_vars=['experiment'], var_name='model', value_name='pehe')

# split into in-sample and out-of-sample -> separate column
# in_models = ['TNet_in', 'SNet1_in', 'SNet2_in', 'SNet3_in', 'SNet_in',
#        'PseudoOutcomeNet_DR_in', 'PseudoOutcomeNet_PW_in',
#        'PseudoOutcomeNet_RA_in', 'PseudoOutcomeNet_RA_S2_in']
# out_models = ['TNet_out',
#        'SNet1_out', 'SNet2_out', 'SNet3_out', 'SNet_out',
#        'PseudoOutcomeNet_DR_out', 'PseudoOutcomeNet_PW_out',
#        'PseudoOutcomeNet_RA_out', 'PseudoOutcomeNet_RA_S2_out']
results.loc[results['model'].str.contains("_in"), 'dataset'] = 'In-Sample'
results.loc[results['model'].str.contains("_out"), 'dataset'] = 'Hold-out'
# remove _in and _out in name
results.loc[results['model'].str.contains("_in"), 'model'] = results.loc[results['model'].str.contains("_in"), 'model'].str.slice(stop=-3)
results.loc[results['model'].str.contains("_out"), 'model'] = results.loc[results['model'].str.contains("_out"), 'model'].str.slice(stop=-4)

results = results.rename(columns={'pehe': 'PEHE', 'model': 'Model', 'dataset': 'Dataset'})

ax = sns.boxplot(x='Model', y='PEHE', hue='Dataset', data=results)
plt.setp(ax)
plt.legend(loc='upper left')
plt.xticks(rotation=90, size=5)
plt.tight_layout()
# plt.show()
plt.savefig('CATENets/results/ihdp/results_boxplot_10rep.png', dpi=600)
plt.close()

