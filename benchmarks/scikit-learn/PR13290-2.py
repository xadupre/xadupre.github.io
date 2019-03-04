import matplotlib.pyplot as plt
import pandas
name = "../../scikit-learn/results/bench_polynomial_features.csv"
df = pandas.read_csv(name)
df['speedup'] = df['time_0_20_2'] / df['time_current']
plt.close('all')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
for color, degree in zip('rgby', sorted(set(df.degree))):
    subdf = df[df.degree == degree]
    subdf.plot(x="time_0_20_2", y="speedup", logx=True, logy=True,
               kind="scatter", ax=ax, label="d=%d" % degree,
               c=color)
ax.set_xlabel("Time(s) of 0.20.2\n.")
ax.set_ylabel("Speed up compare to 0.20.2")
ax.set_title("Acceleration / original time")
ax.plot([df.time_0_20_2.min(), df.time_0_20_2.max()], [2, 2],
        "--", c="black", label="2x")
ax.legend()
plt.show()
