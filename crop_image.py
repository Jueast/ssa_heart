from PIL import Image
import xmltodict
import os
import sys

IMAGE_DIR = sys.argv[1]
TARGET_DIR = sys.argv[2]
xmllist = [ IMAGE_DIR + x for x in os.listdir(IMAGE_DIR) 
               if os.path.isfile(IMAGE_DIR + x) and x.endswith("xml")]

for xml in xmllist:
    with open(xml) as f:
        xmlstr = f.read()
        d = xmltodict.parse(xmlstr)['annotation']
        imgname = IMAGE_DIR + d['filename']+'.jpg'
        try:
            box = [ int(x) for x in d['object']['bndbox'].values()]
        except Exception as e:
            print(d)
        im = Image.open(imgname)
        im = im.crop(box)
        im = im.resize((120,115))
        im = im.convert('L')
        im.save(TARGET_DIR + d['filename'] + '.jpg')

    
