from PIL import Image
import os
import sys
import numpy as np
from skimage import exposure
from ssa_clustering import *
IMAGE_DIR = sys.argv[1]
TARGET_DIR = sys.argv[2]
imlist = [ IMAGE_DIR + x for x in os.listdir(IMAGE_DIR) 
               if os.path.isfile(IMAGE_DIR + x) and x.endswith("jpg")]
i = 0
for imgname in imlist:
    r, pr, r_max = ssa_clustering(imgname, 40, 2, 3, 0.25, False, 42)
    r_max = exposure.equalize_hist(r_max)
    im = Image.fromarray(r_max * 256.0)
#    im.show()
    im = im.convert('L')
#    im.show()

    i += 1
    im.save(TARGET_DIR + str(i)  + '.jpg', 'jpeg')
    if i % 10 == 0:
        print("Progress: {0}/{1}".format(i, len(imlist)))
    
