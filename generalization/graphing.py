import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

acc = [0.00085, 0.00782, .0034, 0.00192, 0.02976]
eod = [0.03025, 0.01826, .02848, 0.02974, 0.00338]
spd = [0.024025, 0.01148, .0674, 0.06142, 0.0024]

x = [.2, .6, .7, .8, 1]

sns.set(font_scale=1.25)


data = (acc, eod, spd)

X = np.arange(5)

a = plt.bar(X + 0.00, data[0], color = 'b', width = 0.25)
e = plt.bar(X + 0.25, data[1], color = 'g', width = 0.25)
s = plt.bar(X + 0.50, data[2], color = 'r', width = 0.25)

plt.xlabel('Total Variation (e-2)')
plt.ylabel('Generalization Gap')
plt.legend(labels=['Acc.', 'EOD', 'SPD'])
plt.xticks(X + 0.25, x)
plt.yticks([0, .01, .02, .03, .04, .05, .06, .07, .08, .09, .1])
plt.tight_layout()
plt.show()

x_new = np.linspace(0, 1)
bspline = interpolate.make_interp_spline(x, acc)
y_new = bspline(x_new)

plt.plot(x, acc, 'b-', label='Accuracy')
#plt.plot(x_new, y_new, 'b-', label='Accuracy')
plt.plot(x, eod, 'r-', label='EOD')
plt.plot(x, spd, 'g-', label='SPD')
plt.legend()
plt.xlabel('Total Variation (e-2)')
plt.ylabel('Generalization Gap')
plt.xlim([0.1,1.1])

plt.tight_layout()
plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
#
#
# def rgama(a):
#     d = a - 1. / 3.
#     c = 1. / np.sqrt(9. * d)
#
#     while True:
#         x = None
#         v = -1
#
#         while v <= 0:
#             x = np.random.normal(0, 1)
#             v = 1. + c * x
#
#         v = np.power(v, 3)
#         u = np.random.uniform()
#
#         if u < 1 - 0.0331 * (x * x) * (x * x):
#             return d * v
#
#         if np.log(u) < 0.5 * x * x + d * (1 - v + np.log(v)):
#             return d * v
#
#
# def rdirch(alphas):
#     k = len(alphas)
#     x = np.array([rgama(alphas[i]) for i in range(k)])
#     total = np.sum(x)
#     x = [s / total for s in x]
#     return x
#
#
# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# ax.set_xlim([0, 1])
# ax.set_title(r'$\operatorname{Dir}(1, 2)$')
# x = np.array([rdirch(np.array([1, 5])) for _ in range(90)])
# _ = sns.distplot(x[:, 0], ax=ax, label=r'$\alpha_1$')
# _ = sns.distplot(x[:, 1], ax=ax, label=r'$\alpha_2$')
# _ = plt.legend()
# plt.show()
#
# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# ax.set_xlim([0, 1])
# ax.set_title(r'$\operatorname{Dir}(2, 4)$')
# x = np.array([rdirch(np.array([1, 1])) for _ in range(10)])
# _ = sns.distplot(x[:, 0], ax=ax, label=r'$\alpha_1$')
# _ = sns.distplot(x[:, 1], ax=ax, label=r'$\alpha_2$')
# _ = plt.legend()
# plt.show()