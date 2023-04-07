import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

Apparatus_volume = float(input("Введите объем аппарата м.куб: "))
Fluid_speed = float(input("Введите скорость потока м.куб/cек: "))
Start_concentration = float(input("Введите начальную концентрацию на входе в реактор кмоль/м.куб:  "))
Reactor_concentration = float(input("Введите начальную концентрацию внутри реактора кмоль/м.куб:  "))


dt = 0.1
E = 0.0001
Reactor_time = Apparatus_volume/Fluid_speed
c = [0]
t = [0]

c[0] = Reactor_concentration
t[0] = 0
i = 0

while True:
    i = i+1
    t.append(t[i-1] + dt)
    print(t[i])
    c.append(c[i-1] + dt * (Start_concentration - c[i - 1]) / Reactor_time)
    print(c[i])

    if abs(c[i]-c[i-1]) < E:
        break
plt.title("Зависимость концентрации от времени в реакторе")
plt.xlabel("Время, секунды")
plt.ylabel("концентрацияб кмоль/м.куб")
plt.plot(t,c)
plt.show()