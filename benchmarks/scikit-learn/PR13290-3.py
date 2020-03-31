import matplotlib.pyplot as plt
import pandas
name = "../../scikit-learn/results/bench_polynomial_features.csv"
df = pandas.read_csv(name)
df = df[df.degree == 2].copy()
df['speedup'] = df['time_0_20_2'] / df['time_current']
plt.close('all')
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

for color, nf in zip('rgby', sorted(set(df.nfeat))):
    subdf = df[df.nfeat == nf]
    subdf.plot(x="time_0_20_2", y="speedup", logx=True, logy=True,
               kind="scatter", ax=ax[0], label="nf=%d" % nf,
               c=color)
ax[0].set_xlabel("Time(s) of 0.20.2\n.")
ax[0].set_ylabel("Speed up compared to 0.20.2")
ax[0].set_title("Acceleration / original time\ndegree == 2")
ax[0].plot([df.time_0_20_2.min(), df.time_0_20_2.max()], [2, 2],
           "--", c="black", label="2x")
ax[0].plot([df.time_0_20_2.min(), df.time_0_20_2.max()], [1, 1],
           "-", c="black", label="1x")
ax[0].legend()

for color, n in zip('rgbycm', sorted(set(df.n))):
    subdf = df[df.n == n]
    subdf.plot(x="time_0_20_2", y="speedup", logx=True, logy=True,
               kind="scatter", ax=ax[1], label="nobs=%d" % n,
               c=color)
ax[1].set_xlabel("Time(s) of 0.20.2\n.")
ax[1].set_ylabel("Speed up compared to 0.20.2")
ax[1].set_title("Acceleration / original time\ndegree == 2")
ax[1].plot([df.time_0_20_2.min(), df.time_0_20_2.max()], [2, 2],
           "--", c="black", label="2x")
ax[1].plot([df.time_0_20_2.min(), df.time_0_20_2.max()], [1, 1],
           "-", c="black", label="1x")
ax[1].legend()
plt.show()
