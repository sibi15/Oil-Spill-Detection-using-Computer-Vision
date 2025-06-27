import tensorflow as tf
import numpy as np

def dice_coefficient(y_true, y_pred):
    smooth = 1e-7
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def bce_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

def dice_bce_mc_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    bce = bce_loss(y_true, y_pred)
    return dice + bce

def dice_mc_metric(y_true, y_pred):
    return dice_coefficient(y_true, y_pred)

# Register custom loss function with TensorFlow
@tf.keras.utils.register_keras_serializable()
class CustomDiceBCELoss(tf.keras.losses.Loss):
    def __init__(self, name='custom_dice_bce_loss', **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, y_true, y_pred):
        return dice_bce_mc_loss(y_true, y_pred)

# Register custom objects
custom_objects = {
    'dice_coefficient': dice_coefficient,
    'dice_loss': dice_loss,
    'bce_loss': bce_loss,
    'dice_bce_mc_loss': dice_bce_mc_loss,
    'dice_mc_metric': dice_mc_metric,
    'CustomDiceBCELoss': CustomDiceBCELoss
}
