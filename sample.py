# c.f. https://dev.pages.lis-lab.fr/ltfatpy/ltfatpy.gabor.html

import numpy as np
import matplotlib.pyplot as plt
from ltfatpy import plotdgt
from ltfatpy import pherm
a = 10
M = 40
L = a * M
h, _ = pherm(L, 4)  # 4th order hermite function.
c = dgt(h, 'gauss', a, M)[0]
# Simple plot: The squared modulus of the coefficients on
# a linear scale
_ = plt.imshow(np.abs(c)**2, interpolation='nearest', origin='upper')
plt.show()
# Better plot: zero-frequency is displayed in the middle,
# and the coefficients are show on a logarithmic scale.
_ = plotdgt(c, a, dynrange=50)
plt.show()
