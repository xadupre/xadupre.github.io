
.. _l-onnxruntime-dt:

Prediction time scikit-learn / onnxruntime: decision tree
=========================================================

.. index:: onnxruntime, decision tree

.. contents::
    :local:

Code
++++

`bench_plot_onnxruntime_decision_tree.py <https://github.com/sdpython/_benchmarks/blob/master/onnx/bench_plot_onnxruntime_decision_tree.py>`_

Overview
++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    name = "../../onnx/results/bench_plot_onnxruntime_decision_tree.csv"
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
    ax[0, 0].plot([df.time_skl.min(), df.time_skl.max()], [0.5, 0.5],
               "--", c="black", label="1/2x")
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
    ax[0, 1].plot([df.time_skl.min(), df.time_skl.max()], [0.5, 0.5],
               "--", c="black", label="1/2x")
    ax[0, 1].plot([df.time_skl.min(), df.time_skl.max()], [1, 1],
               "-", c="black", label="1x")
    ax[0, 1].plot([df.time_skl.min(), df.time_skl.max()], [2, 2],
               "--", c="black", label="2x")
    ax[0, 1].legend()

    # depth
    for color, max_depth in zip('rgbymc', sorted(set(df.max_depth))):
        subdf = df[df.max_depth == max_depth]
        subdf.plot(x="time_skl", y="speedup", logx=True, logy=True,
                   kind="scatter", ax=ax[1, 1], label="mdepth=%d" % max_depth,
                   c=color)
    ax[1, 1].set_xlabel("Time(s) of scikit-learn\n.")
    ax[1, 1].set_ylabel("Speed up compare to scikit-learn")
    ax[1, 1].plot([df.time_skl.min(), df.time_skl.max()], [0.5, 0.5],
               "--", c="black", label="1/2x")
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
    ax[1, 0].plot([df.time_skl.min(), df.time_skl.max()], [0.5, 0.5],
               "--", c="black", label="1/2x")
    ax[1, 0].plot([df.time_skl.min(), df.time_skl.max()], [1, 1],
               "-", c="black", label="1x")
    ax[1, 0].plot([df.time_skl.min(), df.time_skl.max()], [2, 2],
               "--", c="black", label="2x")
    ax[1, 0].legend()

    plt.suptitle("Acceleration onnxruntime / scikit-learn for RandomForest")
    plt.show()

:epkg:`onnxruntime` is always faster in that particular scenario.

Raw results
+++++++++++

:download:`bench_plot_onnxruntime_decision_tree.csv <../../onnx/results/bench_plot_onnxruntime_decision_tree.csv>`

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:

    from pyquickhelper.pandashelper import df2rst
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_decision_tree.csv")
    df = pandas.read_csv(name)
    df['speedup'] = df['time_skl'] / df['time_ort']
    print(df2rst(df, number_format=4))

Detailed graphs
+++++++++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    name = "../../onnx/results/bench_plot_onnxruntime_decision_tree.csv"
    df = pandas.read_csv(name)
    plt.close('all')

    nrows = max(len(set(df.max_depth)) * len(set(df.n_obs)), 2)
    ncols = max(len(set(df.method)), 2)
    fig, ax = plt.subplots(nrows, ncols,
                           figsize=(ncols * 4, nrows * 4))
    pos = 0
    row = 0
    for n_obs in sorted(set(df.n_obs)):
        for max_depth in sorted(set(df.max_depth)):
            pos = 0
            for method in sorted(set(df.method)):
                a = ax[row, pos]
                if row == ax.shape[0] - 1:
                    a.set_xlabel("N features", fontsize='x-small')
                if pos == 0:
                    a.set_ylabel("Time (s) n_obs={}\nmax_depth={}".format(n_obs, max_depth),
                                 fontsize='x-small')

                color = 'b'
                subset = df[(df.method == method) & (df.n_obs == n_obs)
                            & (df.max_depth == max_depth)]
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

    plt.suptitle("Benchmark for DecisionTree sklearn/onnxruntime", fontsize=16)

Benchmark code
++++++++++++++

.. literalinclude:: ../../onnx/bench_plot_onnxruntime_decision_tree.py
    :language: python

Configuration
+++++++++++++

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:

    from pyquickhelper.pandashelper import df2rst
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_decision_tree.time.csv")
    df = pandas.read_csv(name)
    print(df2rst(df, number_format=4))

