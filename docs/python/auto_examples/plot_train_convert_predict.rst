.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_train_convert_predict.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_train_convert_predict.py:


.. _l-logreg-example:

Train, convert and predict with ONNX Runtime
============================================

This example demonstrates an end to end scenario
starting with the training of a machine learned model
to its use in its converted from.

.. contents::
    :local:

Train a logistic regression
+++++++++++++++++++++++++++

The first step consists in retrieving the iris datset.


.. code-block:: default


    from sklearn.datasets import load_iris
    iris = load_iris()
    X, y = iris.data, iris.target

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)








Then we fit a model.


.. code-block:: default


    from sklearn.linear_model import LogisticRegression
    clr = LogisticRegression()
    clr.fit(X_train, y_train)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    C:\Python372_x64\lib\site-packages\sklearn\linear_model\_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)

    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)



We compute the prediction on the test set
and we show the confusion matrix.


.. code-block:: default

    from sklearn.metrics import confusion_matrix

    pred = clr.predict(X_test)
    print(confusion_matrix(y_test, pred))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[14  0  0]
     [ 0 10  2]
     [ 0  0 12]]




Conversion to ONNX format
+++++++++++++++++++++++++

We use module 
`sklearn-onnx <https://github.com/onnx/sklearn-onnx>`_
to convert the model into ONNX format.


.. code-block:: default


    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [('float_input', FloatTensorType([None, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)
    with open("logreg_iris.onnx", "wb") as f:
        f.write(onx.SerializeToString())








We load the model with ONNX Runtime and look at
its input and output.


.. code-block:: default


    import onnxruntime as rt
    sess = rt.InferenceSession("logreg_iris.onnx")

    print("input name='{}' and shape={}".format(
        sess.get_inputs()[0].name, sess.get_inputs()[0].shape))
    print("output name='{}' and shape={}".format(
        sess.get_outputs()[0].name, sess.get_outputs()[0].shape))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    input name='float_input' and shape=[None, 4]
    output name='output_label' and shape=[1]




We compute the predictions.


.. code-block:: default


    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    import numpy
    pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]
    print(confusion_matrix(pred, pred_onx))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[14  0  0]
     [ 0 10  0]
     [ 0  0 14]]




The prediction are perfectly identical.

Probabilities
+++++++++++++

Probabilities are needed to compute other
relevant metrics such as the ROC Curve.
Let's see how to get them first with
scikit-learn.


.. code-block:: default


    prob_sklearn = clr.predict_proba(X_test)
    print(prob_sklearn[:3])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[3.16563550e-03 4.16054556e-01 5.80779808e-01]
     [1.49437471e-03 3.80474536e-01 6.18031090e-01]
     [9.68529455e-01 3.14704190e-02 1.25923161e-07]]




And then with ONNX Runtime.
The probabilies appear to be 


.. code-block:: default


    prob_name = sess.get_outputs()[1].name
    prob_rt = sess.run([prob_name], {input_name: X_test.astype(numpy.float32)})[0]

    import pprint
    pprint.pprint(prob_rt[0:3])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [{0: 0.003165630856528878, 1: 0.4160541892051697, 2: 0.5807801485061646},
     {0: 0.0014943728456273675, 1: 0.3804742097854614, 2: 0.6180314421653748},
     {0: 0.968529462814331, 1: 0.03147042542695999, 2: 1.25923278915252e-07}]




Let's benchmark.


.. code-block:: default

    from timeit import Timer

    def speed(inst, number=10, repeat=20):
        timer = Timer(inst, globals=globals())
        raw = numpy.array(timer.repeat(repeat, number=number))
        ave = raw.sum() / len(raw) / number
        mi, ma = raw.min() / number, raw.max() / number
        print("Average %1.3g min=%1.3g max=%1.3g" % (ave, mi, ma))
        return ave

    print("Execution time for clr.predict")
    speed("clr.predict(X_test)")

    print("Execution time for ONNX Runtime")
    speed("sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]")





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Execution time for clr.predict
    Average 6.04e-05 min=5.3e-05 max=8.35e-05
    Execution time for ONNX Runtime
    Average 3.54e-05 min=3.43e-05 max=4.53e-05

    3.544350000002083e-05



Let's benchmark a scenario similar to what a webservice
experiences: the model has to do one prediction at a time
as opposed to a batch of prediction.


.. code-block:: default


    def loop(X_test, fct, n=None):
        nrow = X_test.shape[0]
        if n is None:
            n = nrow
        for i in range(0, n):
            im = i % nrow
            fct(X_test[im: im+1])

    print("Execution time for clr.predict")
    speed("loop(X_test, clr.predict, 100)")

    def sess_predict(x):
        return sess.run([label_name], {input_name: x.astype(numpy.float32)})[0]

    print("Execution time for sess_predict")
    speed("loop(X_test, sess_predict, 100)")





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Execution time for clr.predict
    Average 0.00408 min=0.0037 max=0.00524
    Execution time for sess_predict
    Average 0.00137 min=0.00125 max=0.00195

    0.0013671394999999896



Let's do the same for the probabilities.


.. code-block:: default


    print("Execution time for predict_proba")
    speed("loop(X_test, clr.predict_proba, 100)")

    def sess_predict_proba(x):
        return sess.run([prob_name], {input_name: x.astype(numpy.float32)})[0]

    print("Execution time for sess_predict_proba")
    speed("loop(X_test, sess_predict_proba, 100)")





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Execution time for predict_proba
    Average 0.00586 min=0.00563 max=0.00646
    Execution time for sess_predict_proba
    Average 0.00146 min=0.00132 max=0.00195

    0.0014575085000000242



This second comparison is better as 
ONNX Runtime, in this experience,
computes the label and the probabilities
in every case.

Benchmark with RandomForest
+++++++++++++++++++++++++++

We first train and save a model in ONNX format.


.. code-block:: default

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    initial_type = [('float_input', FloatTensorType([1, 4]))]
    onx = convert_sklearn(rf, initial_types=initial_type)
    with open("rf_iris.onnx", "wb") as f:
        f.write(onx.SerializeToString())








We compare.


.. code-block:: default


    sess = rt.InferenceSession("rf_iris.onnx")

    def sess_predict_proba_rf(x):
        return sess.run([prob_name], {input_name: x.astype(numpy.float32)})[0]

    print("Execution time for predict_proba")
    speed("loop(X_test, rf.predict_proba, 100)")

    print("Execution time for sess_predict_proba")
    speed("loop(X_test, sess_predict_proba_rf, 100)")





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Execution time for predict_proba
    Average 0.478 min=0.453 max=0.515
    Execution time for sess_predict_proba
    Average 0.00213 min=0.00206 max=0.00229

    0.002129688499999958



Let's see with different number of trees.


.. code-block:: default


    measures = []

    for n_trees in range(5, 51, 5):   
        print(n_trees)
        rf = RandomForestClassifier(n_estimators=n_trees)
        rf.fit(X_train, y_train)
        initial_type = [('float_input', FloatTensorType([1, 4]))]
        onx = convert_sklearn(rf, initial_types=initial_type)
        with open("rf_iris_%d.onnx" % n_trees, "wb") as f:
            f.write(onx.SerializeToString())
        sess = rt.InferenceSession("rf_iris_%d.onnx" % n_trees)
        def sess_predict_proba_loop(x):
            return sess.run([prob_name], {input_name: x.astype(numpy.float32)})[0]
        tsk = speed("loop(X_test, rf.predict_proba, 100)", number=5, repeat=5)
        trt = speed("loop(X_test, sess_predict_proba_loop, 100)", number=5, repeat=5)
        measures.append({'n_trees': n_trees, 'sklearn': tsk, 'rt': trt})

    from pandas import DataFrame
    df = DataFrame(measures)
    ax = df.plot(x="n_trees", y="sklearn", label="scikit-learn", c="blue", logy=True)
    df.plot(x="n_trees", y="rt", label="onnxruntime",
                    ax=ax, c="green", logy=True)
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Prediction time (s)")
    ax.set_title("Speed comparison between scikit-learn and ONNX Runtime\nFor a random forest on Iris dataset")
    ax.legend()



.. image:: /auto_examples/images/sphx_glr_plot_train_convert_predict_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    5
    Average 0.0427 min=0.0403 max=0.0486
    Average 0.00134 min=0.00127 max=0.00142
    10
    Average 0.0653 min=0.0637 max=0.0675
    Average 0.00136 min=0.00128 max=0.00138
    15
    Average 0.0902 min=0.0854 max=0.103
    Average 0.00134 min=0.00129 max=0.00142
    20
    Average 0.11 min=0.106 max=0.112
    Average 0.00173 min=0.00165 max=0.0018
    25
    Average 0.132 min=0.128 max=0.14
    Average 0.00135 min=0.00132 max=0.00137
    30
    Average 0.15 min=0.149 max=0.151
    Average 0.00136 min=0.00134 max=0.00139
    35
    Average 0.173 min=0.171 max=0.175
    Average 0.00137 min=0.00135 max=0.00139
    40
    Average 0.198 min=0.195 max=0.203
    Average 0.0014 min=0.00139 max=0.00142
    45
    Average 0.226 min=0.215 max=0.238
    Average 0.00161 min=0.00143 max=0.00172
    50
    Average 0.238 min=0.236 max=0.244
    Average 0.00152 min=0.00143 max=0.00172

    <matplotlib.legend.Legend object at 0x000001A8A113C048>




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 2 minutes  15.994 seconds)


.. _sphx_glr_download_auto_examples_plot_train_convert_predict.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_train_convert_predict.py <plot_train_convert_predict.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_train_convert_predict.ipynb <plot_train_convert_predict.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
