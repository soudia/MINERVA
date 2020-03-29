import os
import csv
import math
import time
import json
import pickle
import numpy as np
import urllib.request as url
import tensorflow as tf
import scipy

from collections import defaultdict

def pdist(tensor, metric="euclidean"):
    assert isinstance(tensor, (tf.Variable, tf.Tensor,)), "tensor_utils.pdist: Input must be a `tensorflow.Tensor` instance."

    if len(tensor.shape.as_list()) != 2:
        raise ValueError('tensor_utils.pdist: A 2-d tensor must be passed.')

    if metric == "euclidean":

        def pairwise_euclidean_distance(tensor):
            def euclidean_distance(tensor1, tensor2):
                return tf.norm(tensor1 - tensor2)

            m = tensor.shape.as_list()[0]

            distances = []
            for i in range(m):
                for j in range(i + 1, m):
                    distances.append(euclidean_distance(tensor[i], tensor[j]))
            return tf.convert_to_tensor(distances)

        metric_function = pairwise_euclidean_distance
    else:
        raise NotImplementedError(
            "tensor_utils.pdist: "
            "Metric '{metric}' currently not supported!".format(metric=metric)

        )

    return metric_function(tensor)

def _is_vector(tensor):
    return len(tensor.shape.as_list()) == 1

def median(tensor):
    tensor_reshaped = tf.reshape(tensor, [-1])
    n_elements = tensor_reshaped.get_shape()[0]
    sorted_tensor = tf.nn.top_k(tensor_reshaped, n_elements, sorted=True)
    mid_index = n_elements // 2
    if n_elements % 2 == 1:
        return sorted_tensor.values[mid_index]
    return (sorted_tensor.values[mid_index - 1] + sorted_tensor.values[mid_index]) / 2


def squareform(tensor):
    assert isinstance(tensor, tf.Tensor), "tensor_utils.squareform: Input must be a `tensorflow.Tensor` instance."

    tensor_shape = tensor.shape.as_list()
    n_elements = tensor_shape[0]

    if _is_vector(tensor):
        # vector to matrix
        if n_elements == 0:
            return tf.zeros((1, 1), dtype=tensor.dtype)

        # Grab the closest value to the square root of the number
        # of elements times 2 to see if the number of elements is
        # indeed a binomial coefficient
        dimension = int(np.ceil(np.sqrt(n_elements * 2)))

        # Check that `tensor` is of valid dimensions
        if dimension * (dimension - 1) != n_elements * 2:
            raise ValueError(
                "Incompatible vector size. It must be a binomial "
                "coefficient n choose 2 for some integer n >=2."
            )

        n_total_elements_matrix = dimension ** 2

        # Stitch together an upper triangular matrix for our redundant
        # distance matrix from our condensed distance tensor and
        # two tensors filled with zeros.

        n_diagonal_zeros = dimension
        n_fill_zeros = n_total_elements_matrix - n_elements - n_diagonal_zeros

        condensed_distance_tensor = tf.reshape(tensor, shape=(n_elements, 1))
        diagonal_zeros = tf.zeros(
            shape=(n_diagonal_zeros, 1), dtype=condensed_distance_tensor.dtype
        )
        fill_zeros = tf.zeros(
            shape=(n_fill_zeros, 1), dtype=condensed_distance_tensor.dtype
        )
        def upper_triangular_indices(dimension):
            """ For a square matrix with shape (`dimension`, `dimension`),
                return a list of indices into a vector with
                `dimension * dimension` elements that correspond to its
                upper triangular part after reshaping.
            Parameters
            ----------
            dimension : int
                Target dimensionality of the square matrix we want to
                obtain by reshaping a `dimension * dimension` element
                vector.
            Yields
            -------
            index: int
                Indices are indices into a `dimension * dimension` element
                vector that correspond to the upper triangular part of the
                matrix obtained by reshaping it into shape
                `(dimension, dimension)`.
            """
            assert dimension > 0, "tensor_utils.upper_triangular_indices: Dimension must be positive integer!"

            for row in range(dimension):
                for column in range(row + 1, dimension):
                    element_index = dimension * row + column
                    yield element_index

        # General Idea: Use that redundant distance matrices are symmetric:
        # First construct only an upper triangular part and fill
        # everything else with zeros.
        # To the resulting matrix add its transpose, which results in a full
        # redundant distance matrix.

        all_indices = set(range(n_total_elements_matrix))
        diagonal_indices = list(range(0, n_total_elements_matrix, dimension + 1))
        upper_triangular = list(upper_triangular_indices(dimension))

        remaining_indices = all_indices.difference(
            set(diagonal_indices).union(upper_triangular)
        )

        data = (
            # diagonal zeros of our redundant distance matrix
            diagonal_zeros,
            # upper triangular part of our redundant distance matrix
            condensed_distance_tensor,
            # fill zeros for lower triangular part
            fill_zeros
        )

        indices = (
            tuple(diagonal_indices),
            tuple(upper_triangular),
            tuple(remaining_indices)
        )

        stitch_vector = tf.dynamic_stitch(data=data, indices=indices)

        # reshape into matrix
        upper_triangular = tf.reshape(stitch_vector, (dimension, dimension))

        # redundant distance matrices are symmetric
        lower_triangular = tf.transpose(upper_triangular)

        return upper_triangular + lower_triangular
    else:
        raise NotImplementedError(
            "tensor_utils.squareform: Only 1-d "
            "(vector) input is supported!"
        )   


def download(src_filename, saveto, keys):
    ''' Download from urls 
        Args:
            src_filename: text file containing urls
            saveto: folder where to save downloaded files
            keys: dict storing indices and extension   
    '''
    ext = keys['ext']
    i, j = keys['name'], keys['url']
    sep = keys.get('delimiter', '\t')
    saveto = saveto if saveto.endswith('/') else saveto + '/'
    with open(src_filename) as handle:
        links = csv.reader(handle, delimiter=sep)
        for l in links:
            try:
                url.urlretrieve(l[j], saveto + l[i].strip() + ext)
            except:
                continue

def get_image(image_path, image_size, is_crop=True):
    return transform(imread(image_path), image_size, is_crop)

def transform(image, npx=64, is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def imread(path):
    readimage = scipy.misc.imread(path, mode="RGB").astype(np.float)
    return readimage

def merge_channel(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = int(idx / size[1])
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def ims(name, img, num_rows=None):
    if not num_rows:
        img = img[:num_rows, :, :]
    # print img[:10][:10]
    scipy.misc.toimage(img, cmin=0, cmax=1).save(name)


def git_hash_str(hash_len=7):
    import subprocess
    hash_str = subprocess.check_output(['git','rev-parse','HEAD'])
    return str(hash_str[:hash_len])

def normalize(inp, activation, reuse, scope, norm="None"):
    if norm == 'batch_norm':
        return tf.keras.layers.BatchNormalization(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif norm == 'layer_norm':
        return tf.keras.layers.LayerNormalization(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif norm == 'None':
        return activation(inp)

def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID', residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    if max_pool_pad:
        conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    else:
        conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
    normed = normalize(conv_output, activation, reuse, scope)
    if max_pool_pad:
        normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
    return normed