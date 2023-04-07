import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

#init param
nx = 50
ny = 50
nt = 100
xmin = 0
xmax = 2
ymin = 0
ymax = 2

dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)

# init cond
p = np.zeros((ny, nx))
pd = np.zeros((ny, nx))
b = np.zeros((ny, nx))
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)

# source
b[int(ny / 4), int(nx / 4)] = 100
b[int(3 * ny / 4), int(3 * nx / 4)] = -100

# main loop
for i in range(nt):

    pd = p.copy()

    p[1:-1, 1:-1] = (((pd[1:-1, 2:] + pd[1:-1, :-2]) * dy**2 +
            (pd[2:, 1:-1] + pd[:-2, 1:-1]) * dx**2 -
            b[1:-1, 1:-1] * dx**2 * dy**2) /
           (2 * (dx**2 + dy**2)))

    p[0, :] = 0
    p[ny - 1, :] = 0
    p[:, 0] = 0
    p[:, nx - 1] = 0


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

#fig = plt.figure()
#ax = fig.gca(projection='3d')

X, Y = np.meshgrid(x, y)

surf = ax.plot_surface(X, Y, p[:], cmap=cm.viridis)

plt.show()






