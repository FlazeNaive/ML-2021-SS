import numpy as np
import matplotlib.pyplot as plt

E1_corr = np.load("E1_test_corr.npy")
E1_loss = np.load("E1_test_loss.npy")
subscript = [i for i in range(len(E1_corr))]

plt.plot(subscript, E1_corr, color = "deepskyblue")
plt.plot(subscript, E1_loss, color = "firebrick")
plt.show()