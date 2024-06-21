import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('inverted_pendulum.csv', delimiter=',')

states = data[:3]
actions = data[-1]
velocities = states[-1]

fig, ax = plt.subplots()
ax.set_xlabel('iter')
ax.set_ylabel('V')
ax.plot([i for i in range(len(velocities))], velocities, label='ang_vel')
ax.plot([i for i in range(len(velocities))], actions, label='action')
plt.legend()
plt.savefig('inverted_pendulum.png')
