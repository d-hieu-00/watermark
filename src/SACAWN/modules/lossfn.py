import tensorflow as tf
from tensorflow.python.keras import losses

class SACAWNLoss(losses.Loss):
    def __init__(self, impWeight=1.0, robWeight=2.0, extWeight=1.5):
        super().__init__(name="SACAWNLoss")
        self.impWeight = impWeight
        self.robWeight = robWeight
        self.extWeight = extWeight
        self.scc = losses.SparseCategoricalCrossentropy(from_logits=False)

    @staticmethod
    def charErrorRate(y_true, y_pred_probabilities):
        """
        Calculates the character error rate.
        Args:
            y_true (tf.Tensor): True watermark character IDs, shape (B, L).
            y_pred_probabilities (tf.Tensor): Predicted watermark probabilities, shape (B, L, S).
        Returns:
            tf.Tensor: Scalar character error rate (0 = perfect).
        """
        # Convert predicted probabilities to predicted character IDs
        # This is the same logic as the get_predicted_char_ids function in the Canvas.
        y_pred_ids = tf.argmax(y_pred_probabilities, axis=-1, output_type=tf.int32)

        # Ensure y_true is also int32 for comparison
        y_true_ids = tf.cast(y_true, tf.int32)

        # Calculate where true and predicted IDs are not equal
        errors = tf.cast(tf.not_equal(y_true_ids, y_pred_ids), tf.float32)

        # The mean of errors across all characters and batch items
        # A value of 0 means perfect extraction.
        return tf.reduce_mean(errors)

# return tf.argmax(predicted_probabilities, axis=-1, output_type=tf.int32)
    def __call__(self, img_orig, img_watermarked, wm_true, wm_pred):
        """
        img_original:       (B, H, W, 3)
        img_watermarked:    (B, H, W, 3)
        wm_true: (B, L)
        wm_pred: (B, L, S)
        """

        # imperceptibility
        L_imp = 1.0 - tf.reduce_mean(tf.image.ssim(img_orig, img_watermarked, max_val=1.0)) # Lower is better

        # robustness
        L_rob = self.scc(wm_true, wm_pred)

        # extraction accuracy
        L_ext = SACAWNLoss.charErrorRate(wm_true, wm_pred)

        total_loss = (self.impWeight * L_imp +
                      self.robWeight * L_rob +
                      self.extWeight * L_ext)

        return total_loss, L_imp, L_rob, L_ext
