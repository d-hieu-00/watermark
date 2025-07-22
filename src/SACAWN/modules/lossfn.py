import tensorflow as tf
from tensorflow.python.keras import losses

class SACAWNLoss(losses.Loss):
    def __init__(self, impWeight=1.0, robWeight=2.0, extWeight=1.5):
        super().__init__(name="SACAWNLoss")
        self.impWeight = impWeight
        self.robWeight = robWeight
        self.extWeight = extWeight
        self.bce = losses.BinaryCrossentropy(from_logits=True)
        self.ce  = losses.SparseCategoricalCrossentropy(from_logits=True)

    def __call__(self, img_orig, img_watermarked, wm_true, wm_pred):
        # one-hot wms
        wm_true_onehot = tf.one_hot(tf.cast(wm_true, dtype=tf.int32), depth=wm_true.shape[0])
        wm_pred_onehot = tf.one_hot(tf.cast(wm_pred, dtype=tf.int32), depth=wm_pred.shape[0])

        # imperceptibility
        ssim = tf.image.ssim(img_orig, img_watermarked, max_val=1.0)
        L_imp = 1.0 - tf.reduce_mean(ssim) # Lower is better

        # robustness (bit-level)
        L_rob = self.bce(wm_true_onehot, wm_pred_onehot) # Lower is better

        # extraction accuracy (symbol-level)
        L_ext = self.ce(wm_true_onehot, wm_pred_onehot) # Lower is better

        total_loss = (self.impWeight * L_imp +
                      self.robWeight * L_rob +
                      self.extWeight * L_ext)

        return total_loss, L_imp, L_rob, L_ext
