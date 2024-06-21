import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('inverted_pendulum.csv', delimiter=',')

states     = data[:4]
actions    = data[-1]
poses      = states[0]
angles     = states[1]
cart_vs    = states[2]
velocities = states[3]
iters = [i for i in range(len(velocities))]

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))

ax0.set_xlabel('iter')
ax0.plot(iters, velocities/4, label='ang_vel')
ax0.plot(iters, actions, label='action')
ax0.plot(iters, angles, label='angle')
ax0.legend()

ax1.set_xlabel('iter')
ax1.plot(iters, poses, label='poses')
ax1.plot(iters, cart_vs/3, label='cart_vs')
ax1.legend()

plt.savefig('inverted_pendulum.png')
