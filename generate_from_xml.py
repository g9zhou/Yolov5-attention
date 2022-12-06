import xml.etree.ElementTree as ET
import os
from glob import glob

XML_PATH = './stanford_dogset/Annotation/'
CLASSES_PATH = './classes.txt'
TXT_PATH = './stanford_dogset/annobad.txt'


'''loads the classes'''
def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    class_names = [c.split()[1] for c in class_names]
    return class_names


classes = get_classes(CLASSES_PATH)
assert len(classes) > 0, 'no class names detected!'
print(f'num classes: {len(classes)}')

# output file
list_file = open(TXT_PATH, 'w')
for dir in os.listdir("./stanford_dogset/Annotation"):
    if dir[0] == '.' or dir[-3:] == 'txt': continue
    for path in os.listdir(XML_PATH+dir):
        in_file = open(XML_PATH+dir+'/'+path)

        # Parse .xml file
        tree = ET.parse(in_file)
        root = tree.getroot()
        # Write object information to .txt file
        file_name = root.find('filename').text
        list_file.write(file_name+'.jpg')
        for obj in root.iter('object'):
            cls = obj.find('name').text 
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        list_file.write('\n')
list_file.close()