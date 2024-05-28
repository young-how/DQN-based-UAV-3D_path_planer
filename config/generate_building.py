import xml.etree.ElementTree as ET
import random

def generate_random_house(x_range, y_range, z_range, r_range, h_range):
    threaten = ET.Element("Threaten")
    
    threaten_type = ET.SubElement(threaten, "Threaten_Type")
    threaten_type.text = "building"
    
    position = ET.SubElement(threaten, "position")
    x = ET.SubElement(position, "x")
    x.text = str(random.uniform(*x_range))
    y = ET.SubElement(position, "y")
    y.text = str(random.uniform(*y_range))
    z = ET.SubElement(position, "z")
    z.text = str(random.uniform(*z_range))
    
    r = ET.SubElement(threaten, "_R")
    r.text = str(random.uniform(*r_range))
    
    h = ET.SubElement(threaten, "_H")
    h.text = str(random.uniform(*h_range))
    
    return threaten

def generate_houses_xml(num_houses, x_range, y_range, z_range, r_range, h_range, output_file):
    root = ET.Element("buildings")
    
    for _ in range(num_houses):
        house = generate_random_house(x_range, y_range, z_range, r_range, h_range)
        root.append(house)
    
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)

if __name__ == "__main__":
    num_houses = 26  # 自定义生成房屋的数量
    x_range = (0, 500)  # 自定义x的取值范围
    y_range = (0, 500)  # 自定义y的取值范围
    z_range = (0, 0)  # 自定义z的取值范围
    r_range = (10, 50)  # 自定义_R的取值范围
    h_range = (10, 50)  # 自定义_H的取值范围
    output_file = "./config/buildings.xml"  # 输出XML文件的名称

    generate_houses_xml(num_houses, x_range, y_range, z_range, r_range, h_range, output_file)
