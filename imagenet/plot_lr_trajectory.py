import matplotlib.pyplot as plt
import numpy as np

# set epoch as a numpy arrany of ints ranging from 0-5
epoch = np.arange(0, 5)
print(epoch)
args_lr = 0.1
lr_change = 1


# plot the learning rate trajectory
lr = args_lr * (0.1 ** (epoch // lr_change))

plt.plot(epoch, lr)
plt.xlabel('Epoch')
plt.ylabel('Learning rate')
plt.title('Learning rate trajectory')
# save plot to file
plt.savefig('plot_lr_trajectory.png')

print(lr)