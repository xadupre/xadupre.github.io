import os
import pandas
import matplotlib.pyplot as plt

full_name = os.path.normpath(os.path.abspath(
    os.path.join("..", "..", "inference_profiling.csv")))
df = pandas.read_csv(full_name)

# but a graph is usually better...
gr_dur = df[['dur', "args_op_name"]].groupby("args_op_name").sum().sort_values('dur')
gr_n = df[['dur', "args_op_name"]].groupby("args_op_name").count().sort_values('dur')
gr_n = gr_n.loc[gr_dur.index, :]

fig, ax = plt.subplots(1, 3, figsize=(8, 4))
gr_dur.plot.barh(ax=ax[0])
gr_dur /= gr_dur['dur'].sum()
gr_dur.plot.barh(ax=ax[1])
gr_n.plot.barh(ax=ax[2])
ax[0].set_title("duration")
ax[1].set_title("proportion")
ax[2].set_title("n occurences");
for a in ax:
    a.legend().set_visible(False)

plt.show()