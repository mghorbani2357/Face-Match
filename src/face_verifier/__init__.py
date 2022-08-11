import numpy as np
import tensorflow as tf


class ArcFace:
    __model = tf.keras.models.load_model('arch_face.h5')
    __verification_threshold = 4.4

    def match(self, face1, face2):
        distance = self.euclidean_distance(self.encode_image(face1), self.encode_image(face2))
        return distance < self.__verification_threshold, distance

    @staticmethod
    def euclidean_distance(source_representation, test_representation):
        distance = source_representation - test_representation
        distance = np.sum(np.multiply(distance, distance))
        distance = np.sqrt(distance)
        return distance

    def encode_image(self, image):
        return self.__model.predict(image)[0]

    # def preprocess_face(img, target_size=(112, 112)):
    #     img = cv2.imread(img)
    #     img = detect_face(img)
    #     img = cv2.resize(img, target_size)
    #     img_pixels = image.img_to_array(img)
    #     img_pixels = np.expand_dims(img_pixels, axis=0)
    #     img_pixels /= 255
    #     return img_pixels


model = ArcFace()
