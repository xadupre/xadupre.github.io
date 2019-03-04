import matplotlib.pyplot as plt
import pandas
name = "../../onnx/results/bench_plot_onnxruntime_random_forest.csv"
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
ax[0, 0].plot([df.time_skl.min(), df.time_skl.max()], [10, 10],
              "--", c="black", label="10x")
ax[0, 0].legend()

# estimators
for color, n_estimators in zip('rgbymc', sorted(set(df.n_estimators))):
    subdf = df[df.n_estimators == n_estimators]
    subdf.plot(x="time_skl", y="speedup", logx=True, logy=True,
               kind="scatter", ax=ax[0, 1], label="n_est=%d" % n_estimators,
               c=color)
ax[0, 1].set_title("Acceleration / original time")
ax[0, 1].plot([df.time_skl.min(), df.time_skl.max()], [1, 1],
              "-", c="black", label="1x")
ax[0, 1].plot([df.time_skl.min(), df.time_skl.max()], [2, 2],
              "--", c="black", label="2x")
ax[0, 1].plot([df.time_skl.min(), df.time_skl.max()], [10, 10],
              "--", c="black", label="10x")
ax[0, 1].legend()

# depth
for color, max_depth in zip('rgbymc', sorted(set(df.max_depth))):
    subdf = df[df.max_depth == max_depth]
    subdf.plot(x="time_skl", y="speedup", logx=True, logy=True,
               kind="scatter", ax=ax[1, 1], label="mdepth=%d" % max_depth,
               c=color)
ax[1, 1].set_xlabel("Time(s) of scikit-learn\n.")
ax[1, 1].set_ylabel("Speed up compare to scikit-learn")
ax[1, 1].plot([df.time_skl.min(), df.time_skl.max()], [1, 1],
              "-", c="black", label="1x")
ax[1, 1].plot([df.time_skl.min(), df.time_skl.max()], [2, 2],
              "--", c="black", label="2x")
ax[1, 1].plot([df.time_skl.min(), df.time_skl.max()], [10, 10],
              "--", c="black", label="10x")
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
ax[1, 0].plot([df.time_skl.min(), df.time_skl.max()], [10, 10],
              "--", c="black", label="10x")
ax[1, 0].legend()

plt.suptitle("Acceleration onnxruntime / scikit-learn for RandomForest")
plt.show()
