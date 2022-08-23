# import seaborn as sns
# import matplotlib.pyplot as plt

# acc = [.0121, .09628, .077445, .131979]
# eod = [.11106, .03842, .04918, .07342]
# spd = [.01786, .0163, .01232, .01372]
# x = [.14, .16, .34, .48]
#
# sns.set(font_scale=1.25)
#
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
#
# ax1.plot(x, acc, 'b-*', label='Accuracy')
# ax1.plot(x, eod, 'r-o', label='EOD')
# ax2.plot(x, spd, 'g-+', label='SPD')
# ax1.legend()
# ax1.set_xlabel('Total Variation (e-3)')
# ax1.set_ylabel('Generalization Gap (Acc/EOD)')
# ax1.set_xlim([.10,.5])
# ax2.set_ylabel('Generalization Gap (SPD)')
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
x = np.array([rdirch(np.array([1, 1])) for _ in range(1000)])
_ = sns.distplot(x[:, 0], ax=ax, label=r'$\alpha_1$')
_ = sns.distplot(x[:, 1], ax=ax, label=r'$\alpha_2$')
_ = plt.legend()
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_xlim([0, 1])
ax.set_title(r'$\operatorname{Dir}(2, 4)$')
x = np.array([rdirch(np.array([1, 4])) for _ in range(1000)])
_ = sns.distplot(x[:, 0], ax=ax, label=r'$\alpha_1$')
_ = sns.distplot(x[:, 1], ax=ax, label=r'$\alpha_2$')
_ = plt.legend()
plt.show()