import xmltodict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# 读取XML文件
def read_xml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        xml_content = file.read()
    return xmltodict.parse(xml_content)

# 绘制圆柱体
def plot_cylinder(ax, x, y, z, radius, height, color='blue', alpha=0.5):
    phi = np.linspace(0, 2 * np.pi, 100)
    z_vals = np.linspace(z, z + height, 50)
    phi_grid, z_grid = np.meshgrid(phi, z_vals)
    x_grid = x + radius * np.cos(phi_grid)
    y_grid = y + radius * np.sin(phi_grid)
    
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha)

# 主函数
def main():
    file_path = './config/buildings.xml'
    data = read_xml(file_path)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    buildings = data['buildings']['Threaten']
    if not isinstance(buildings, list):
        buildings = [buildings]
    
    for building in buildings:
        position = building['position']
        x = float(position['x'])
        y = float(position['y'])
        z = float(position['z'])
        radius = float(building['_R'])
        height = float(building['_H'])
        
        plot_cylinder(ax, x, y, z, radius, height)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

if __name__ == "__main__":
    main()
