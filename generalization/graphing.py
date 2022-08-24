# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
#
# acc = [.1362, .1148, .1517, .1924, .1746, .1645, .2628, .3267, .2598]
# eod = [.0457, .0277, .1243, .0366, .1256, .0256, .0906, .0634, .1118]
# spd = [.0077, .0621, .0127, .0017, .0371, .0201, .0138, .1017, .0994]
# x = [.05, .09, .13, .16, .2, .21, .46, .57, .61]
#
# sns.set(font_scale=1.25)
#
#
# data = (acc, eod, spd)
#
# X = np.arange(9)
#
# plt.bar(X + 0.00, data[0], color = 'b', width = 0.25)
# plt.bar(X + 0.25, data[1], color = 'g', width = 0.25)
# plt.bar(X + 0.50, data[2], color = 'r', width = 0.25)
#
# plt.xlabel('Total Variation (e-3)')
# plt.ylabel('Generalization Gap')
# plt.legend(labels=['Acc', 'EOD', 'SPD'])
# plt.xticks(X + 0.25, x)
# plt.tight_layout()
# plt.show()
#
# e_x = np.polyfit(x, eod, 1)
# s_x = np.polyfit(x, eod, 1)
# trend_eod = np.poly1d(e_x)
# trend_spd = np.poly1d(s_x)
#
# plt.plot(x, acc, 'b-', label='Accuracy')
# plt.plot(x, eod, 'r-', label='EOD')
# plt.plot(x, trend_eod(eod), 'r--')
# plt.plot(x, spd, 'g-', label='SPD')
# plt.plot(x, trend_spd(spd), 'g--')
# plt.legend()
# plt.xlabel('Total Variation (e-3)')
# plt.ylabel('Generalization Gap')
# plt.xlim([0,.65])
#
# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def rgama(a):
    d = a - 1. / 3.
    c = 1. / np.sqrt(9. * d)

    while True:
        x = None
        v = -1

        while v <= 0:
            x = np.random.normal(0, 1)
            v = 1. + c * x

        v = np.power(v, 3)
        u = np.random.uniform()

        if u < 1 - 0.0331 * (x * x) * (x * x):
            return d * v

        if np.log(u) < 0.5 * x * x + d * (1 - v + np.log(v)):
            return d * v


def rdirch(alphas):
    k = len(alphas)
    x = np.array([rgama(alphas[i]) for i in range(k)])
    total = np.sum(x)
    x = [s / total for s in x]
    return x


fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_xlim([0, 1])
ax.set_title(r'$\operatorname{Dir}(1, 2)$')
x = np.array([rdirch(np.array([1, 5])) for _ in range(90)])
_ = sns.distplot(x[:, 0], ax=ax, label=r'$\alpha_1$')
_ = sns.distplot(x[:, 1], ax=ax, label=r'$\alpha_2$')
_ = plt.legend()
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_xlim([0, 1])
ax.set_title(r'$\operatorname{Dir}(2, 4)$')
x = np.array([rdirch(np.array([1, 1])) for _ in range(10)])
_ = sns.distplot(x[:, 0], ax=ax, label=r'$\alpha_1$')
_ = sns.distplot(x[:, 1], ax=ax, label=r'$\alpha_2$')
_ = plt.legend()
plt.show()