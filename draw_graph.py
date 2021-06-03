import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

E1_corr = np.load("E1_test_corr.npy")
E12_corr = np.load("E1+E2_test_corr.npy")
subscript = [i for i in range(len(E1_corr))]

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('epoch')
ax1.set_ylabel('E1 Accuracy', color=color)
ax1.plot(subscript, E1_corr, color=color)
ax1.tick_params(axis='y', labelcolor=color)

color = 'tab:red'
ax1.set_ylabel('E1+E2 Accuracy', color=color)
ax1.plot(subscript, E12_corr, color=color)
ax1.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title("Accuracy")
plt.savefig('ACC.jpg', bbox_inches='tight')
quit() 

PATT = "E1"

E1_corr = np.load(PATT+"_test_corr.npy")
E1_loss = np.load(PATT+"_test_loss.npy")
subscript = [i for i in range(len(E1_corr))]

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy', color=color)
ax1.plot(subscript, E1_corr, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # 创建共用x轴的第二个y轴

color = 'tab:red'
ax2.set_ylabel('loss', color=color)
ax2.plot(subscript, E1_loss, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title(PATT)
plt.savefig(PATT+'.jpg', bbox_inches='tight')
# plt.show()