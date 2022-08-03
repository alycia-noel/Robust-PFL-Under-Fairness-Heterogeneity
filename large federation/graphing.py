import seaborn as sns
import matplotlib.pyplot as plt

acc = [.7872, .7689, .7694, .771, .7696, .767, .7667, .7762,.7737, .7737]
eod = [-.0164, .0414, .1163, .1287, .1295, .0738, .2371, .1207, .1005, .095]
spd = [-.0262, .0539, -.0466, -.0313, .0069, -.0545, -.0127, -.0176, -.0387, -.0631]

x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

sns.set()

plt.plot(x, acc, 'b-', label='Accuracy')
plt.plot(x, eod, 'r-o', label='EOD')
plt.plot(x, spd, 'g-+', label='SPD')
plt.legend(loc='right')
plt.title('Accuracy, EOD, and SPD for Varying Number of Clients')
plt.xlabel('Num Clients')
plt.ylabel('Accuracy/EOD/SPD')
plt.tight_layout()
plt.show()