python -m plots.statistic_autocorrelation \
  glow svhn_working --id_datasets svhn \
  --datasets celeba svhn \
  --anomaly_detection gradients_L2_norms --batch_size 5 \
  --max_lag 300


python -m plots.statistic_autocorrelation \
  diffusion diffusion_svhn_1_timesteps --id_datasets svhn \
  --datasets celeba svhn \
  --anomaly_detection gradients_L2_norms --batch_size 5 \
  --max_lag 100 # smaller lag because our diffusion models have fewer layers