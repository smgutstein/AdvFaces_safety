import argparse
import foolbox
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg
from tensorflow_vgg.vgg16 import Vgg16


def get_sample_image(image_name,shape=(224, 224),
                     data_format='channels_last'):
    """ Returns an example image and its imagenet class label.

    Parameters
    ----------
    shape : list of integers
        The shape of the returned image.
    data_format : str
        "channels_first" or "channels_last"

    Returns
    -------
    image : array_like
        The example image.

    label : int
        The imagenet label associated with the image.

    NOTE: This function is deprecated and will be removed in the future.
    """
    assert len(shape) == 2
    assert data_format in ['channels_first', 'channels_last']

    path = os.path.join(os.path.dirname(__file__), image_name)
    image = Image.open(path)
    image = image.resize(shape)
    image = np.asarray(image, dtype=np.float32)
    image = image[:, :, :3]
    assert image.shape == shape + (3,)
    if data_format == 'channels_first':
        image = np.transpose(image, (2, 0, 1))
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test foolbox")
    parser.add_argument("-i", "--image", type=str,
                        default = "./test_data/kitten2.png")

    args = vars(parser.parse_args())
    image = get_sample_image(args["image"])
    synset = [l.strip() for l in open("./test_data/synset.txt").readlines()]
                        
    
    images = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))

    logits2 = Vgg16()
    logits2.build(images)
    

    with foolbox.models.TensorFlowModel(images, logits2.fc8, (0, 255)) as model:
        #restorer.restore(model.session, '/path/to/vgg_19.ckpt')
        idx = np.argmax(model.forward_one(image))
        raw_conf = model.forward_one(image)
        raw_max = raw_conf.max()
        raw_conf2 = raw_conf-raw_max
        raw_conf2_exp = np.exp(raw_conf2)
        raw_conf_norm = 1./raw_conf2_exp.sum()
        conf = raw_conf2_exp * raw_conf_norm
        category = ' '.join(synset[idx].split()[1:])

        print(idx, category, conf[idx])
