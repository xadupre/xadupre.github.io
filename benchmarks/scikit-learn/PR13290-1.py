import matplotlib.pyplot as plt
import pandas
name = "../../scikit-learn/results/bench_polynomial_features.csv"
df = pandas.read_csv(name)
plt.close('all')

nrows = len(set(df.degree))
fig, ax = plt.subplots(nrows, 4, figsize=(nrows * 4, 12))
pos = 0

for di, degree in enumerate(sorted(set(df.degree))):
    pos = 0
    for order in sorted(set(df.order)):
        for interaction_only in sorted(set(df.interaction_only)):
            a = ax[di, pos]
            if di == ax.shape[0] - 1:
                a.set_xlabel("N observations", fontsize='x-small')
            if pos == 0:
                a.set_ylabel("Time (s) degree={}".format(degree),
                             fontsize='x-small')

            for color, nfeat in zip('brgyc', sorted(set(df.nfeat))):
                subset = df[(df.degree == degree) & (df.nfeat == nfeat) &
                            (df.interaction_only == interaction_only) &
                            (df.order == order)]
                if subset.shape[0] == 0:
                    continue
                subset = subset.sort_values("n")
                label = "nf={} l=0.20.2".format(nfeat)
                subset.plot(x="n", y="time_0_20_2", label=label, ax=a,
                            logx=True, logy=True, c=color, style='--')
                label = "nf={} l=now".format(nfeat)
                subset.plot(x="n", y="time_current", label=label, ax=a,
                            logx=True, logy=True, c=color)

            a.legend(loc=0, fontsize='x-small')
            if di == 0:
                a.set_title("order={} interaction_only={}".format(
                    order, interaction_only), fontsize='x-small')
            pos += 1

plt.suptitle("Benchmark for PolynomialFeatures")
plt.show()
