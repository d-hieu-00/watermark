import logging
import tensorflow as tf
from keras import losses

logger = logging.getLogger(__name__)

class SACAWNLoss(losses.Loss):
    def __init__(self, impWeight=1.0, robWeight=2.0, extWeight=1.5):
        super().__init__(name="SACAWNLoss")
        self.impWeight = impWeight
        self.robWeight = robWeight
        self.extWeight = extWeight
        self.bce = losses.BinaryCrossentropy(from_logits=False)
        self.cce = losses.CategoricalCrossentropy(from_logits=False)
        logger.info(f"Setup SACAWNLoss with impWeight: {impWeight}, robWeight: {robWeight}, extWeight: {extWeight}")

    @staticmethod
    def extractionAccuracy(wm_true, wm_pred):
        """
        Calculates the character error rate.
        Args:
            wm_true (tf.Tensor): True watermark character IDs, shape (B, L, S).
            wm_pred (tf.Tensor): Predicted watermark probabilities, shape (B, L, S).
        Returns:
            tf.Tensor: Scalar character error rate (0 = perfect).
        """
        wm_true_int = tf.argmax(wm_true, axis=-1)
        wm_pred_int = tf.argmax(wm_pred, axis=-1)

        # Calculate where true and predicted IDs are not equal
        errors = tf.cast(tf.not_equal(wm_true_int, wm_pred_int), tf.float32)

        # A value of 0 means perfect extraction.
        return tf.reduce_mean(errors)

    @staticmethod
    def imperceptibilityLoss(img_orig, img_watermarked):
        img_orig = tf.clip_by_value(img_orig, 0.0, 1.0)
        img_watermarked = tf.clip_by_value(img_watermarked, 0.0, 1.0)

        # L1 Pixel Loss
        L_pix = tf.reduce_mean(tf.abs(img_orig - img_watermarked))

        # SSIM Loss
        L_ssim = 1.0 - tf.reduce_mean(tf.image.ssim(img_orig, img_watermarked, max_val=1.0))

        # Combine (weighted)
        L_imp = 0.2 * L_pix + 0.8 * L_ssim
        return L_imp

    def L_imp(self, img_orig, img_watermarked):
        return SACAWNLoss.imperceptibilityLoss(img_orig, img_watermarked)

    def L_rob(self, wm_true, wm_pred):
        return 0.5 * self.bce(wm_true, wm_pred) + 0.5 * self.cce(wm_true, wm_pred)

    def L_ext(self, wm_true, wm_pred):
        return SACAWNLoss.extractionAccuracy(wm_true, wm_pred)

    def L_total(self, L_imp, L_rob, L_ext):
        total_loss = (self.impWeight * L_imp +
                      self.robWeight * L_rob +
                      self.extWeight * L_ext)

        return total_loss, L_imp, L_rob, L_ext

    def __call__(self, img_orig, img_watermarked, wm_true, wm_pred):
        """
        img_original:       (B, H, W, 3)
        img_watermarked:    (B, H, W, 3)
        wm_true: (B, L, S)
        wm_pred: (B, L, S)
        """

        # imperceptibility
        L_imp = self.L_imp(img_orig, img_watermarked)

        # robustness
        L_rob = self.L_rob(wm_true, wm_pred)

        # extraction accuracy
        L_ext = self.L_ext(wm_true, wm_pred)

        return self.L_total(L_imp, L_rob, L_ext)
