import textwrap
import matplotlib.pyplot as plt
from yobx.doc import plot_text

sample = textwrap.dedent(
    """
    --- a/foo.py
    +++ b/foo.py
    @@ -1,3 +1,3 @@
    def foo():
    -    return 1
    +    return 2
    """)
ax = plot_text(
    sample,
    title="sample diff",
    line_color_map={"+": "#2a9d2a", "-": "#cc2222", "@": "#1a6fbf"},
)
plt.show()