import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

E1_corr = np.load("E1_test_corr.npy")
E1_loss = np.load("E1_test_loss.npy")
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
plt.title("E1")
plt.savefig('E1.jpg', bbox_inches='tight')
# plt.show()