from yobx.doc import plot_dot
from yobx.helpers.einsum_helper import decompose_einsum

model = decompose_einsum("bij,bjk->bik", (2, 3, 4), (2, 4, 5))
plot_dot(model)