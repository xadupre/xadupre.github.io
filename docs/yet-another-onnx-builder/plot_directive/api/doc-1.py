import matplotlib.pyplot as plt
import onnx.parser
from yobx.doc import plot_dot

model = onnx.parser.parse_model(
    '''
    <ir_version: 8, opset_import: [ "": 18]>
    agraph (float[N] x) => (float[N] z) {
        two = Constant <value_float=2.0> ()
        four = Add(two, two)
        z = Mul(four, four)
    }
''')

ax = plot_dot(model)
ax.set_title("Dummy graph")
plt.show()