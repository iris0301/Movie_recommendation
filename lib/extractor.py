import tensorflow as tf
import urllib
from PIL import Image
import numpy as np
import IPython
import matplotlib.pyplot as plt



class ImgExtractor(object):
    def __init__(self, model='VGG16', *args, **kwargs):
        self.models = [
            'VGG16', 'VGG19', 'ResNet50',
            'DenseNet121', 'DeseNet169', 'inception_v3',
            'inceptionResNetV2'
        ]
        if model == "VGG16":
            self.model = tf.keras.applications.VGG16(*args, **kwargs)
        elif model == "VGG19":
            self.model = tf.keras.applications.VGG19(*args, **kwargs)
        elif model == "Resnet50":
            self.model = tf.keras.applications.ResNet50(*args, **kwargs)
        elif model == "DenseNet121":
            self.model = tf.keras.applications.DenseNet121(*args, **kwargs)
        elif model == "DeseNet169":
            self.model = tf.keras.applications.DeseNet169(*args, **kwargs)
        elif model == "inception_v3":
            self.model = tf.keras.applications.inception_v3(*args, **kwargs)
        elif model == "inceptionResNetV2":
            self.model = tf.keras.applications.inceptionResNetV2(*args, **kwargs)
        else:
            print("you must select one model from {}".format(self.models))

    def get_features(self, img_path, **kwargs):
        raw_arr = self.get_raw_arr(img_path)

        img_arr = np.expand_dims(raw_arr, axis=0) #(224,224,3,None)

        features = self.model.predict(img_arr)
        return features

    def get_raw_arr(self, img_path, *args, **kwargs):
        if img_path.startswith("http"):
            print("load Image from url")
            html_response = urllib.request.urlopen(img_path)
            raw_img = Image.open(html_response)
            resized_img = raw_img.resize((224, 224))
            raw_arr = np.array(resized_img)[:, :, 0:3]

            return raw_arr

        else:
            print('load Image locally')

            # img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224), *args, **kwargs)
            # raw_arr = tf.keras.preprocessing.image.img_to_array(img)
            raw_img = Image.open(img_path)
            resized_img = raw_img.resize((224, 224))
            raw_arr = np.array(resized_img)[:, :, 0:3]

        return raw_arr

    def arr_2_features(self, img_arr, **kwargs):
        if img_arr.shape == (224, 224, 3):
            img_arr = np.expand_dims(img_arr, axis=0)
        elif img_arr == (1, 224, 224, 3):
            pass
        else:
            raise Exception("shape of input arr should be (1,224,224,3) or (224,224,3)")

        features = self.model.predict(img_arr)
        return features

    def show_img(self, img_arr, **kwargss):
        plt.imshow(img_arr, aspect="auto")
        plt.show()