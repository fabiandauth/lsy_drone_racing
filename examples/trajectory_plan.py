import yaml
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt


class Gate:
    def __init__(self, x, y, type, yaw, width = 0.375, height= 0.375, frame = 0.05):
        self.x = x
        self.y = y
        self.type = type
        if self.type == 0:
            self.z = 0.5
        else:
            self.z = 1 
        self.yaw = yaw
        self.width = width
        self.height = height
        self.frame = frame

    def get_position(self):
        return self.x, self.y, self.z

    def get_boundaries(self):
        middle_point = np.array([self.x, self.y, self.z])
        c1 = middle_point + np.array([-self.width/2, 0, -self.height/2])
        c2 = middle_point + np.array([-self.width/2, 0, self.height/2])
        c3 = middle_point + np.array([self.width/2, 0, self.height/2])
        c4 = middle_point + np.array([self.width/2, 0, -self.height/2])
        corners = [c1, c2, c3, c4]

        def _rotate_corners(corners, yaw):
            rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                       [np.sin(yaw), np.cos(yaw), 0],
                                       [0, 0, 1]])
            rotated_corners = []
            for corner in corners:
                rotated_corner = np.dot(rotation_matrix, corner)
                rotated_corners.append(rotated_corner)
            return rotated_corners
            
        
        self.corners = _rotate_corners(corners, self.yaw)


    def intersects(self, point):
        pass
        
    def plot_gate_3D(self, ax):
        self.get_boundaries()
        gate = Poly3DCollection([self.corners])
        ax.add_collection3d(gate)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        
    
    
class Obstacle:
    def __init__(self, x, y, height = 1.05, radius = 0.05):
        self.x = x
        self.y = y
        self.height = height
        self.radius = radius

    def get_position(self):
        return self.x, self.y

    def get_dimensions(self):
        return self.width, self.height, self.depth
    

gate1 = Gate(0.45, -1, 1, 2.35)
gate2 = Gate(1.0, -1.55, 0, -0.78)
gate3 = Gate(0.0, 0.5, 1, 1)
gate4 = Gate(-0.5, -1, 0, 3.14)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

gate1.plot_gate_3D(ax)
gate2.plot_gate_3D(ax)
gate3.plot_gate_3D(ax)
gate4.plot_gate_3D(ax)

plt.show()