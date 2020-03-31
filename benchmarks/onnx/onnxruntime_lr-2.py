import matplotlib.pyplot as plt
import pandas
name = "../../onnx/results/bench_plot_onnxruntime_logreg.csv"
df = pandas.read_csv(name)
plt.close('all')

nrows = max(len(set(df.fit_intercept)) * len(set(df.n_obs)), 2)
ncols = max(len(set(df.method)), 2)
fig, ax = plt.subplots(nrows, ncols,
                       figsize=(ncols * 4, nrows * 4))
pos = 0
row = 0
for n_obs in sorted(set(df.n_obs)):
    for fit_intercept in sorted(set(df.fit_intercept)):
        pos = 0
        for method in sorted(set(df.method)):
            a = ax[row, pos]
            if row == ax.shape[0] - 1:
                a.set_xlabel("N features", fontsize='x-small')
            if pos == 0:
                a.set_ylabel("Time (s) n_obs={}\nfit_intercept={}".format(n_obs, fit_intercept),
                             fontsize='x-small')

            color = 'b'
            subset = df[(df.method == method) & (df.n_obs == n_obs) &
                        (df.fit_intercept == fit_intercept)]
            if subset.shape[0] == 0:
                continue
            subset = subset.sort_values("nfeat")
            label = "skl"
            subset.plot(x="nfeat", y="time_skl", label=label, ax=a,
                        logx=True, logy=True, c=color, style='--')
            label = "ort"
            subset.plot(x="nfeat", y="time_ort", label=label, ax=a,
                        logx=True, logy=True, c=color)

            a.legend(loc=0, fontsize='x-small')
            if row == 0:
                a.set_title("method={}".format(method), fontsize='x-small')
            pos += 1
        row += 1

plt.suptitle(
    "Benchmark for LogisticRegression sklearn/onnxruntime", fontsize=16)
