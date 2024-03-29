# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# pylint: disable=C0411,C0412,C0413

"""

.. _l-logreg-example-speed:

Train, convert and predict with ONNX Runtime
============================================

This example demonstrates an end to end scenario
starting with the training of a machine learned model
to its use in its converted from.

Train a logistic regression
+++++++++++++++++++++++++++

The first step consists in retrieving the iris datset.
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

####################################
# Then we fit a model.

clr = LogisticRegression()
clr.fit(X_train, y_train)

####################################
# We compute the prediction on the test set
# and we show the confusion matrix.
from sklearn.metrics import confusion_matrix

pred = clr.predict(X_test)
print(confusion_matrix(y_test, pred))

####################################
# Conversion to ONNX format
# +++++++++++++++++++++++++
#
# We use module
# `sklearn-onnx <https://github.com/onnx/sklearn-onnx>`_
# to convert the model into ONNX format.

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [("float_input", FloatTensorType([None, 4]))]
onx = convert_sklearn(clr, initial_types=initial_type)
with open("logreg_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

##################################
# We load the model with ONNX Runtime and look at
# its input and output.

import onnxruntime as rt

sess = rt.InferenceSession("logreg_iris.onnx", providers=rt.get_available_providers())

print("input name='{}' and shape={}".format(sess.get_inputs()[0].name, sess.get_inputs()[0].shape))
print("output name='{}' and shape={}".format(sess.get_outputs()[0].name, sess.get_outputs()[0].shape))

##################################
# We compute the predictions.

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

import numpy

pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]
print(confusion_matrix(pred, pred_onx))

###################################
# The prediction are perfectly identical.
#
# Probabilities
# +++++++++++++
#
# Probabilities are needed to compute other
# relevant metrics such as the ROC Curve.
# Let's see how to get them first with
# scikit-learn.

prob_sklearn = clr.predict_proba(X_test)
print(prob_sklearn[:3])

#############################
# And then with ONNX Runtime.
# The probabilies appear to be

prob_name = sess.get_outputs()[1].name
prob_rt = sess.run([prob_name], {input_name: X_test.astype(numpy.float32)})[0]

import pprint

pprint.pprint(prob_rt[0:3])

###############################
# Let's benchmark.
from timeit import Timer


def speed(inst, number=5, repeat=10):
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

###############################
# Let's benchmark a scenario similar to what a webservice
# experiences: the model has to do one prediction at a time
# as opposed to a batch of prediction.


def loop(X_test, fct, n=None):
    nrow = X_test.shape[0]
    if n is None:
        n = nrow
    for i in range(0, n):
        im = i % nrow
        fct(X_test[im : im + 1])


print("Execution time for clr.predict")
speed("loop(X_test, clr.predict, 50)")


def sess_predict(x):
    return sess.run([label_name], {input_name: x.astype(numpy.float32)})[0]


print("Execution time for sess_predict")
speed("loop(X_test, sess_predict, 50)")

#####################################
# Let's do the same for the probabilities.

print("Execution time for predict_proba")
speed("loop(X_test, clr.predict_proba, 50)")


def sess_predict_proba(x):
    return sess.run([prob_name], {input_name: x.astype(numpy.float32)})[0]


print("Execution time for sess_predict_proba")
speed("loop(X_test, sess_predict_proba, 50)")

#####################################
# This second comparison is better as
# ONNX Runtime, in this experience,
# computes the label and the probabilities
# in every case.

##########################################
# Benchmark with RandomForest
# +++++++++++++++++++++++++++
#
# We first train and save a model in ONNX format.
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)

initial_type = [("float_input", FloatTensorType([1, 4]))]
onx = convert_sklearn(rf, initial_types=initial_type)
with open("rf_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

###################################
# We compare.

sess = rt.InferenceSession("rf_iris.onnx", providers=rt.get_available_providers())


def sess_predict_proba_rf(x):
    return sess.run([prob_name], {input_name: x.astype(numpy.float32)})[0]


print("Execution time for predict_proba")
speed("loop(X_test, rf.predict_proba, 50)")

print("Execution time for sess_predict_proba")
speed("loop(X_test, sess_predict_proba_rf, 50)")

##################################
# Let's see with different number of trees.

measures = []

for n_trees in range(5, 51, 5):
    print(n_trees)
    rf = RandomForestClassifier(n_estimators=n_trees)
    rf.fit(X_train, y_train)
    initial_type = [("float_input", FloatTensorType([1, 4]))]
    onx = convert_sklearn(rf, initial_types=initial_type)
    with open("rf_iris_%d.onnx" % n_trees, "wb") as f:
        f.write(onx.SerializeToString())
    sess = rt.InferenceSession("rf_iris_%d.onnx" % n_trees, providers=rt.get_available_providers())

    def sess_predict_proba_loop(x):
        return sess.run([prob_name], {input_name: x.astype(numpy.float32)})[0]

    tsk = speed("loop(X_test, rf.predict_proba, 25)", number=5, repeat=4)
    trt = speed("loop(X_test, sess_predict_proba_loop, 25)", number=5, repeat=4)
    measures.append({"n_trees": n_trees, "sklearn": tsk, "rt": trt})

from pandas import DataFrame

df = DataFrame(measures)
ax = df.plot(x="n_trees", y="sklearn", label="scikit-learn", c="blue", logy=True)
df.plot(x="n_trees", y="rt", label="onnxruntime", ax=ax, c="green", logy=True)
ax.set_xlabel("Number of trees")
ax.set_ylabel("Prediction time (s)")
ax.set_title("Speed comparison between scikit-learn and ONNX Runtime\nFor a random forest on Iris dataset")
ax.legend()
