import matplotlib.pyplot as plt


n_timesteps = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] # , 999]

ours_batch_size_5 = \
    [0.8793, 0.8760, 0.8811, 0.8799, 0.8780, 0.8709,
    0.8673, 0.8553, 0.8521,
     0.7719]  # , 0.5723]
typicality_batch_size_5 = \
    [0.8131, 0.8223, 0.8322, 0.8419, 0.8479, 0.8503,
    0.8489, 0.8447, 0.8358,
     0.8187]  #, 0.5584]


# plt.xticks(n_timesteps, n_timesteps)

plt.semilogx(n_timesteps, ours_batch_size_5, label="ours", base=2.)
plt.semilogx(n_timesteps, typicality_batch_size_5, label="typicality", base=2.)

plt.xticks(n_timesteps, n_timesteps)

plt.title("Ablation study on the effect of the number of timesteps on the \n"
          " performance of a diffusion model using batch size 5.")

plt.xlabel("number of timesteps (log scale)")
plt.ylabel("AUROC (averaged over all dataset pairings)")


plt.legend()

plt.show()
