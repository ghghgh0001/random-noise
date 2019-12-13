from keras.applications import ResNet50
import cv2
from glob2 import glob
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications import VGG16
from keras.applications import VGG19


def classify_img(acc):
    m=0
    n=0
    model = ResNet50(weights='imagenet')
    location = r'./data/flower_photos/alpha1.5/' + str(acc) + '/'
    for file1 in glob(location + "*.jpg"):
        # img_path = '/mnt/463G/gh/keras/mine/data/one_pixel/640/0038671.jpg'
        n=n+1
        img = image.load_img(file1, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        # print('Predicted:', decode_predictions(preds, top=5)[0])
        for i in range(5):
            if decode_predictions(preds, top=5)[0][i][1] == 'daisy':
                m = m+1
                # print(m, n)
                break
    return float(m)/float(n)

if __name__ == '__main__':
    acclist = []
    for i in range(3):
        acc = classify_img((i * 2 + 1) / 2)
        acclist.append(acc)
        print(acclist.__len__(), acclist)
        # K.clear_session()
    listText = open('./data/flower_photos/alpha_1.5_list.txt', 'a')
    list_write = 'resnet\n' + str(acclist) + '\n'
    listText.write(list_write)
    listText.close()

    # for i in range(10):
    #     acc = classify_img(i * 5 + 25)
    #     acclist.append(acc)
    #     print(acclist.__len__(), acclist)
    #     # K.clear_session()
    # listText = open('./data/flower_photos/alpha_1.5_list.txt', 'a')
    # list_write = 'VGG16\n' + str(acclist) + '\n'
    # listText.write(list_write)
    # listText.close()

