import matplotlib.pyplot as plt
import pandas
name = "../../scikit-learn/results/bench_plot_polynomial_features_partial_fit.csv"
df = pandas.read_csv(name)
plt.close('all')

nrows = max(len(set(df.nfeat)), 2)
ncols = max(1, 2)
fig, ax = plt.subplots(nrows, ncols,
                       figsize=(nrows * 4, ncols * 4))
colors = "gbry"
row = 0
for nfeat in sorted(set(df.nfeat)):
    pos = 0
    for _ in range(1):
        a = ax[row, pos]
        if row == ax.shape[0] - 1:
            a.set_xlabel("N observations", fontsize='x-small')
        if pos == 0:
            a.set_ylabel("Time (s) nfeat={}".format(nfeat),
                         fontsize='x-small')

        subset = df[df.nfeat == nfeat]
        if subset.shape[0] == 0:
            continue
        subset = subset.sort_values("n_obs")

        label = "SGD"
        subset.plot(x="n_obs", y="time_sgd", label=label, ax=a,
                    logx=True, logy=True, c=colors[0], style='--')
        label = "SGD-SKL"
        subset.plot(x="n_obs", y="time_pipe_skl", label=label, ax=a,
                    logx=True, logy=True, c=colors[1], style='--')
        label = "SGD-FAST"
        subset.plot(x="n_obs", y="time_pipe_fast", label=label, ax=a,
                    logx=True, logy=True, c=colors[2])
        label = "SGD-SLOW"
        subset.plot(x="n_obs", y="time_pipe_slow", label=label, ax=a,
                    logx=True, logy=True, c=colors[3])

        a.legend(loc=0, fontsize='x-small')
        if row == 0:
            a.set_title("--", fontsize='x-small')
        pos += 1
    row += 1

plt.suptitle("Benchmark for Polynomial with SGDClassifier", fontsize=16)
