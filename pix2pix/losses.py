import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target,LAMBDA=100):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    #l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) # default
    # l1_loss = dice_loss(target, gen_output) # train 3
    l1_loss = dice_bce_loss(target, gen_output) # train 2

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss    
    
def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

def dice(y_true, y_pred, smooth=1):
    # y_true = K.cast(y_true,'bool')
    # y_pred = K.cast(y_pred,'bool')
    # y_pred = K.cast(y_pred,'float32')
    # y_true = K.cast(y_true,'float32')
    im_sum = K.sum(y_pred) + K.sum(y_true)
    intersection = y_true * y_pred
    return 2.*K.sum(intersection)/im_sum

def dice_loss(y_true, y_pred, smooth=1):
    return 1-dice(y_true, y_pred)

def dice_bce_loss(y_true, y_pred):
    dice_loss = 1-dice(y_true, y_pred)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    bce_ = bce(y_true, y_pred)
    
    dice_bce_loss = (dice_loss + bce_) / 2
    
    return dice_bce_loss

# IoU
def iou(y_true, y_pred):
    '''intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score'''
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    iou_ = intersection/union
    return iou_

def calc_metric(y_pred,y_true):
    cm = confusion_matrix(y_pred.flatten(),y_true.flatten())
    tn, fp, fn, tp = cm.ravel()
    dice = (2.0 * tp) / ((2.0 * tp) + fp + fn)
    jaccard = (1.0 * tp) / (tp + fp + fn) 
    sensitivity = (1.0 * tp) / (tp + fn)
    specificity = (1.0 * tn) / (tn + fp)
    accuracy = (1.0 * (tn + tp)) / (tn + fp + tp + fn)
    auc = 1 - 0.5 * (((1.0 * fp) / (fp + tn)) + ((1.0 * fn) / (fn + tp)))
    prec = float(tp)/float(tp + fp)
    fscore = float(2*tp)/float(2*tp + fp + fn)
    return dice,jaccard,sensitivity,specificity,accuracy,auc,prec,fscore