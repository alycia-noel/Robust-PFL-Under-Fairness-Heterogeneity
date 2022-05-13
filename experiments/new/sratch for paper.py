import os
from collections import OrderedDict

import numpy
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder

x = numpy.array([random.triangular(100,200) for i in range(250)])
y = numpy.array([-(x[i] ** 1.2) - random.triangular(100,200)* numpy.random.normal(0,2) +1000 for i in range(250)])

x1 = numpy.array([random.triangular(50,150) for i in range(300)])
y1 = numpy.array([-(x1[i] ** 1.2) - random.triangular(50,150)* numpy.random.normal(0,1.5) + 500 for i in range(300)])

x2 = numpy.array([random.triangular(0,100) for i in range(300)])
y2 = numpy.array([-(x2[i] ** 1.2) - random.triangular(0,100)* numpy.random.normal(0,2.5)  for i in range(300)])

# all_x = numpy.append(x, x1)
all_x = numpy.append(x1, x2)
# all_y = numpy.append(y, y1)
all_y = numpy.append(y1, y2)

# plt.scatter(x, y, alpha=.5, c='c')
# z = np.polyfit(x, y, 1)
# p = np.poly1d(z)
# plt.plot(x,p(x),"k--")

plt.scatter(x1, y1, alpha=.5, c='m')
z1 = np.polyfit(x1, y1, 1)
p = np.poly1d(z1)
plt.plot(x1,p(x1),"k--")

plt.scatter(x2, y2, alpha=.5, c='b')
z2 = np.polyfit(x2, y2, 1)
p = np.poly1d(z2)
plt.plot(x2,p(x2),"k--")

z_all = np.polyfit(all_x, all_y, 1)
p = np.poly1d(z_all)
plt.plot(all_x,p(all_x),"r--")
plt.title('Example of Simpson\'s Paradox with Negative Individual Correlation')
plt.show()