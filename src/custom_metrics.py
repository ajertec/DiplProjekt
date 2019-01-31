import keras.backend as K

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def euclidean_distance_loss(y_true, y_pred): 
    """ Euclidean distance loss https://en.wikipedia.org/wiki/Euclidean_distance 
    :param y_true: TensorFlow/Theano tensor 
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true :return: float """ 
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))