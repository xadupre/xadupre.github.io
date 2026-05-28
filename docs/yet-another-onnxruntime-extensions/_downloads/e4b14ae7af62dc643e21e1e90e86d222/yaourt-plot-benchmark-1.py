import pandas
from yaourt.plot._data import hhistograms_data
from yaourt.plot.benchmark import hhistograms

df = pandas.DataFrame(hhistograms_data())
hhistograms(df, keys=("input", "name"))