import cv2
path = "./20191012_0/"
num = 0
path_save = "./pic/"
for i in range(14942):
    if (i==0):
        continue
    if (i % 14 !=0):
        continue
    paths = path + str(i) + ".jpg"
    print(paths)
    image = cv2.imread(paths)
    num = num + 1
    print(image.shape)
    p = path_save + str(i) + ".jpg"
    print(p)
    cv2.imwrite(p, image)

