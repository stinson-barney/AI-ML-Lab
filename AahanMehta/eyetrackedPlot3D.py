import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

file = pd.read_csv(r'C:\Users\Admin\Desktop\colleger work\Semester2\AI&ML Lab\zigzag\Video-30.csv')

x = file.iloc[:,0].values
y = file.iloc[:,1].values


fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X-axis")

ax.set_zlabel("Y-axis")

ax.set_ylabel("time")
ax.plot(x,range(len(x)),y)

plt.savefig(r'C:\Users\Admin\Desktop\colleger work\Semester2\AI&ML Lab\zigzag\Video-30.png')

plt.show()