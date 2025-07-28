import tensorflow as tf
from tensorflow.python.keras import losses

class SACAWNLoss(losses.Loss):
    def __init__(self, impWeight=1.0, robWeight=2.0, extWeight=1.5):
        super().__init__(name="SACAWNLoss")
        self.impWeight = impWeight
        self.robWeight = robWeight
        self.extWeight = extWeight
        self.bce = losses.BinaryCrossentropy(from_logits=False)
        self.cc  = losses.CategoricalCrossentropy(from_logits=True)

    def __call__(self, img_orig, img_watermarked, wm_true, wm_pred):
        # imperceptibilityrobustness
        ssim = tf.image.ssim(img_orig, img_watermarked, max_val=1.0)
        L_imp = 1.0 - tf.reduce_mean(ssim) # Lower is better

        # robustness (bit-level)
        # Ensure wm_pred contains probabilities (not logits) and shapes match
        wm_true_clipped = tf.clip_by_value(wm_true, 0.0, 1.0)
        wm_pred_prob = tf.clip_by_value(wm_pred, 0.0, 1.0)
        if len(wm_pred_prob.shape) != len(wm_true_clipped.shape):
            wm_pred_prob = tf.squeeze(wm_pred_prob)
        L_rob = self.bce(wm_true_clipped, wm_pred_prob) # Lower is better

        # extraction accuracy (symbol-level)
        L_ext = 0 # self.cc(wm_true, wm_pred) # Lower is better

        total_loss = (self.impWeight * L_imp +
                      self.robWeight * L_rob +
                      self.extWeight * L_ext)

        return total_loss, L_imp, L_rob, L_ext
