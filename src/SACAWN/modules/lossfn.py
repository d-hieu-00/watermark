import tensorflow as tf
from tensorflow.python.keras import losses

class SACAWNLoss(losses.Loss):
    def __init__(self, impWeight=1.0, robWeight=2.0, extWeight=1.5):
        super().__init__(name="SACAWNLoss")
        self.impWeight = impWeight
        self.robWeight = robWeight
        self.extWeight = extWeight
        self.bce = losses.BinaryCrossentropy()
        self.ce  = losses.SparseCategoricalCrossentropy()

    def __call__(self, img_orig, img_watermarked, wm_true, wm_pred):
        # imperceptibility
        ssim = tf.image.ssim(img_orig, img_watermarked, max_val=1.0)
        L_imp = 1.0 - tf.reduce_mean(ssim) # Lower is better

        # robustness (bit-level)
        L_rob = self.bce(wm_true, wm_pred) # Lower is better

        # extraction accuracy (symbol-level)
        L_ext = self.ce(wm_true, wm_pred) # Lower is better

        total_loss = (self.impWeight * L_imp +
                      self.robWeight * L_rob +
                      self.extWeight * L_ext)

        return total_loss, L_imp, L_rob, L_ext
