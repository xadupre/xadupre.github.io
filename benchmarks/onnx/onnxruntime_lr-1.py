import matplotlib.pyplot as plt
import pandas
name = "../../onnx/results/bench_plot_onnxruntime_logreg.csv"
df = pandas.read_csv(name)
df['speedup'] = df['time_skl'] / df['time_ort']
plt.close('all')
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# method
for color, method in zip('rgbymc', sorted(set(df.method))):
    subdf = df[df.method == method]
    subdf.plot(x="time_skl", y="speedup", logx=True, logy=True,
               kind="scatter", ax=ax[0, 0], label=method,
               c=color)
ax[0, 0].set_xlabel("Time(s) of scikit-learn\n.")
ax[0, 0].plot([df.time_skl.min(), df.time_skl.max()], [1, 1],
              "-", c="black", label="1x")
ax[0, 0].plot([df.time_skl.min(), df.time_skl.max()], [2, 2],
              "--", c="black", label="2x")
ax[0, 0].legend()

# nobs
for color, n_obs in zip('rgbymc', sorted(set(df.n_obs))):
    subdf = df[df.n_obs == n_obs]
    subdf.plot(x="time_skl", y="speedup", logx=True, logy=True,
               kind="scatter", ax=ax[0, 1], label="n_obs=%d" % n_obs,
               c=color)
ax[0, 1].set_ylabel("Speed up compare to scikit-learn")
ax[0, 1].plot([df.time_skl.min(), df.time_skl.max()], [1, 1],
              "-", c="black", label="1x")
ax[0, 1].plot([df.time_skl.min(), df.time_skl.max()], [2, 2],
              "--", c="black", label="2x")
ax[0, 1].legend()

# fit_intercept
for color, fit_intercept in zip('rgbymc', sorted(set(df.fit_intercept))):
    subdf = df[df.fit_intercept == fit_intercept]
    subdf.plot(x="time_skl", y="speedup", logx=True, logy=True,
               kind="scatter", ax=ax[1, 1], label="fi=%d" % fit_intercept,
               c=color)
ax[1, 1].set_xlabel("Time(s) of scikit-learn\n.")
ax[1, 1].set_ylabel("Speed up compare to scikit-learn")
ax[1, 1].plot([df.time_skl.min(), df.time_skl.max()], [1, 1],
              "-", c="black", label="1x")
ax[1, 1].plot([df.time_skl.min(), df.time_skl.max()], [2, 2],
              "--", c="black", label="2x")
ax[1, 1].legend()

# features
for color, nfeat in zip('rgbymc', sorted(set(df.nfeat))):
    subdf = df[df.nfeat == nfeat]
    subdf.plot(x="time_skl", y="speedup", logx=True, logy=True,
               kind="scatter", ax=ax[1, 0], label="nfeat=%d" % nfeat,
               c=color)
ax[1, 0].set_ylabel("Speed up compare to scikit-learn")
ax[1, 0].plot([df.time_skl.min(), df.time_skl.max()], [1, 1],
              "-", c="black", label="1x")
ax[1, 0].plot([df.time_skl.min(), df.time_skl.max()], [2, 2],
              "--", c="black", label="2x")
ax[1, 0].legend()

plt.suptitle("Acceleration onnxruntime / scikit-learn for RandomForest")
plt.show()
