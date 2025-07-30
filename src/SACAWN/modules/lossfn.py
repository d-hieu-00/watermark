import logging
import tensorflow as tf
from tensorflow.python.keras import losses

logger = logging.getLogger(__name__)

class SACAWNLoss(losses.Loss):
    def __init__(self, impWeight=1.0, robWeight=2.0, extWeight=1.5):
        super().__init__(name="SACAWNLoss")
        self.impWeight = impWeight
        self.robWeight = robWeight
        self.extWeight = extWeight
        self.scc = losses.SparseCategoricalCrossentropy(from_logits=False)
        logger.info(f"Setup SACAWNLoss with impWeight: {impWeight}, robWeight: {robWeight}, extWeight: {extWeight}")

    @staticmethod
    def extractionAccuracy(y_true, y_pred):
        """
        Calculates the character error rate.
        Args:
            y_true (tf.Tensor): True watermark character IDs, shape (B, L).
            y_pred (tf.Tensor): Predicted watermark probabilities, shape (B, L, S).
        Returns:
            tf.Tensor: Scalar character error rate (0 = perfect).
        """
        # Convert predicted probabilities to predicted character IDs
        # This is the same logic as the get_predicted_char_ids function in the Canvas.
        y_pred_ids = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

        # Ensure y_true is also int32 for comparison
        y_true_ids = tf.cast(y_true, tf.int32)

        # Calculate where true and predicted IDs are not equal
        errors = tf.cast(tf.not_equal(y_true_ids, y_pred_ids), tf.float32)

        # The mean of errors across all characters and batch items
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
        return self.scc(wm_true, wm_pred)

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
        wm_true: (B, L)
        wm_pred: (B, L, S)
        """
        # print(f"Shape: img_orig: {img_orig.shape} img_watermarked: {img_watermarked.shape} wm_true: {wm_true.shape} wm_pred: {wm_pred.shape}")

        # imperceptibility
        L_imp = SACAWNLoss.imperceptibilityLoss(img_orig, img_watermarked)

        # robustness
        L_rob = self.scc(wm_true, wm_pred)

        # extraction accuracy
        L_ext = SACAWNLoss.extractionAccuracy(wm_true, wm_pred)

        total_loss = (self.impWeight * L_imp +
                      self.robWeight * L_rob +
                      self.extWeight * L_ext)

        return total_loss, L_imp, L_rob, L_ext
