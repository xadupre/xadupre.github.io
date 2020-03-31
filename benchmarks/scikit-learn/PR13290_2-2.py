import matplotlib.pyplot as plt
import pandas
name = "../../scikit-learn/results/bench_plot_polynomial_features_partial_fit.csv"
df = pandas.read_csv(name)
df['speedup'] = df['time_pipe_slow'] / df['time_pipe_fast']
plt.close('all')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
for color, nfeat in zip('rgby', sorted(set(df.nfeat))):
    subdf = df[df.nfeat == nfeat]
    subdf.plot(x="time_pipe_slow", y="speedup", logx=True, logy=True,
               kind="scatter", ax=ax, label="nf=%d" % nfeat,
               c=color)
ax.set_xlabel("Time(s) of 0.20.2\n.")
ax.set_ylabel("Speed up compare to 0.20.2")
ax.set_title("Acceleration / original time")
ax.plot([df.time_pipe_slow.min(), df.time_pipe_fast.max()], [2, 2],
        "--", c="black", label="2x")
ax.legend()
plt.show()
