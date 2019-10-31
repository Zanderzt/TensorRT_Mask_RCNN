file = "./pic1.txt"
f = open(file)
import cv2
path = "./20191012_0/"
path_save = "./pic/"
lines = f.readlines()
for line in lines:
    line = line.strip()
    paths = path + line + ".jpg"
    print(paths)
    image = cv2.imread(paths)
    p = path_save + line + ".jpg"
    print(p)
    cv2.imwrite(p, image)

f.close()
