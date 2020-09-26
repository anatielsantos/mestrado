import tensorflow.keras.backend as K
import numpy as np

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def dice(y_true, y_pred, smooth=1):
    # y_true = K.cast(y_true,'bool')
    # y_pred = K.cast(y_pred,'bool')
    # y_pred = K.cast(y_pred,'float32')
    # y_true = K.cast(y_true,'float32')
    im_sum = K.sum(y_pred) + K.sum(y_true)
    intersection = y_true * y_pred
    return 2.*K.sum(intersection)/im_sum

def dice_c0(y_true, y_pred, smooth=1):
    return dice(y_true[...,0],y_pred[...,0])

def dice_c1(y_true, y_pred, smooth=1):
    return dice(y_true[...,1],y_pred[...,1])

def dice_c2(y_true, y_pred, smooth=1):
    return dice(y_true[...,2],y_pred[...,2])

def dice_c3(y_true, y_pred, smooth=1):
    return dice(y_true[...,3],y_pred[...,3])

def dice_multiclass(y_true,y_pred,n_classes=4):
    _dice=0.0
    class_weight={0:0.2,1:350.0,2:160.0,3:465.0}
    total_weigth=0
    for i in range(1,n_classes):
        # total_weigth+=class_weight[i]
       _dice+=dice(y_true[...,i],y_pred[...,i])#*class_weight[i]
    # dice /= n_classes #media
    # dice /= total_weigth #media ponderada
    return _dice 

def dice_medio(y_true,y_pred,n_classes=4):
    _dice=0.0
    class_weight={0:0.2,1:350.0,2:160.0,3:465.0}
    total_weigth=0
    for i in range(n_classes):
        # total_weigth+=class_weight[i]
        _dice+=dice(y_true[...,i],y_pred[...,i])
    _dice /= n_classes #media
    # dice /= total_weigth #media ponderada
    return _dice

def weighted_loss(y_true,y_pred):
    from tensorflow.keras.losses import binary_crossentropy
    n_classes=4
    class_weight={0:0.2,1:350.0,2:160.0,3:465.0}
    total=0.0
    for i in range(n_classes):
        # total_weigth+=class_weight[i]
        total+=binary_crossentropy(y_true[...,i],y_pred[...,i])*class_weight[i]
    return total

def dice_multiclass_loss(y_true, y_pred,n_classes=4):
    return 1-dice_multiclass(y_true, y_pred,n_classes)

def dice_loss(y_true, y_pred):
    return 1-dice(y_true, y_pred)


def dice_calc(im1, im2, empty_score=1.0):
    """
    Computes the dice_val coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice_val : float
        dice_val coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        
    Notes
    -----
    The order of inputs for `dice_val` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute dice_val coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum
