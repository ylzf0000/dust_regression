import glob
import os
import cv2 as cv

real_path = os.path.realpath(__file__)
real_dir = real_path[:real_path.rfind('/')]
data_dir = os.path.join(real_dir,'data')
print(real_dir)
img_path = r'/disk/ljli/PycharmProjects/darknet-with-opencv/darknet-master/build/darknet/x64/data/obj'
filenames = glob.glob(os.path.join(img_path, '*.txt'))
dust_count = [0] * 3 # [6616, 5671, 3540] = 15,827
for filename in filenames:
    print(filename)
    with open(filename) as f:
        count = 1
        for line in f.readlines():
            line = line.split()
            class_id = int(line[0])
            if 2 <= class_id <= 4:
                dust_count[class_id - 2] += 1
                img = cv.imread(filename[:-4] + '.jpg')
                print(img.size, img.shape)
                if img is None or img.size == 0:
                    continue
                x, y, w, h = map(float, line[1:])
                x -= w / 2
                y -= h / 2
                x = max(0, int(x * img.shape[1]))
                y = max(0, int(y * img.shape[0]))
                w = min(img.shape[1], int(w * img.shape[1]))
                h = min(img.shape[0], int(h * img.shape[0]))
                print(x, y, w, h)
                img = img[y: y + h, x: x + w, :]
                # print(img.shape)

                img_name = os.path.split(filename)[-1][:-4] + f'_i{count}_c{class_id - 2}'

                img_file = os.path.join(data_dir, img_name + '.jpg')       
                cv.imwrite(img_file, img)

                # label_file = os.path.join(data_dir, img_name + '.txt')
                # with open(label_file, 'w') as label_f:
                #     label_f.write(str(class_id - 2))
                count += 1
                print(img_file)

    # break
print(dust_count)
'''
0 truck
1 lifted truck
2 low dust
3 medium dust
4 high dust
5 opened grab
6 closed grab
'''
