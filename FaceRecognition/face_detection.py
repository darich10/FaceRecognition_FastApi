import skimage
from skimage.transform import resize
from mtcnn.mtcnn import MTCNN
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import io
from fastapi import File

detector = MTCNN()


def read_image(path: str) -> np:
    """
    Read an image by PIL
    :param path: image file path
    :return: image numpy
    """
    image = skimage.io.imread(path)
    return image


def find_face(image: np) -> np:
    """
    Find a face in the image by the mtcnn method
    :param image: Original image
    :return: Face image if exist
    """
    # create the detector, using default weights
    # detect faces in the image
    results = detector.detect_faces(image)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = image[y1:y2, x1:x2]
    return face


def face_detection(im: np):
    """

    :param image:
    :return:
    """
    im = find_face(im)
    im = resize(im, (112, 112))
    # im = preprocessing(im)
    return im


def preprocessing(im: np) -> np:
    """
    Preprocess the image according to
    :param im: numpy array of face image
    :return: numpy array of normalized image
    """
    im -= 127.5
    im /= 128
    return im


def read_bytes_img(image: File) -> np:
    """
    Read a byte image and convert to a numpy array
    :param image: bytes image
    :return: numpy array
    """
    image = np.asarray(Image.open(io.BytesIO(image)))
    return image


if __name__ == "__main__":
    img = read_image("../images/0000001.jpg")
    img = face_detection(img)
    print(img.shape)
    plt.imshow(img, interpolation='nearest')
    plt.show()
