import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

nx = 81
ny = 81
nt = 200
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .2
dt = sigma * dx

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u = np.ones((nx, ny))
un = np.ones((nx, ny))

# initial conditions
u[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2


for n in range(nt + 1):
    un = u.copy()
    row, col = u.shape

    u[1:, 1:] = (un[1:, 1:] - (c * dt / dx * (un[1:, 1:] - un[1:, :-1])) - (c * dt / dy * (un[1:, 1:] - un[:-1, 1:])))

    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1



fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

#fig = plt.figure()
#ax = fig.gca(projection='3d')

X, Y = np.meshgrid(x, y)

surf = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)

plt.show()







