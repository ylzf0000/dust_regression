'''
测试集: 1000
验证集: 1000
训练集: 13827
'''
import glob
import os
import shutil
import numpy as np
real_path = os.path.realpath(__file__)
real_dir = real_path[:real_path.rfind('/')]
filenames = glob.glob('data/*.jpg')
# filenames = np.array(filenames)
np.random.shuffle(filenames)
print(type(filenames))
d = {
    'data_valid': filenames[:1000],
    'data_test': filenames[1000:2000],
    'data_train': filenames[2000:],
}
for name, data in d.items():
    for img in data:
        _img = img[img.rfind('/') + 1:]
        to = os.path.join(name, _img)
        shutil.copy(img, to)
