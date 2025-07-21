import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.losses import BinaryCrossentropy, Loss

from tensorflow import __version__; keras.__version__ = __version__

# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.losses import BinaryCrossentropy, Loss

def GetShape(x):
    return tf.shape(x)

class BaseModel(keras.Model):
    def __init__(self, maxWmLen=256, vocabSize=128):
        """
        Initialize the watermark embedder model.
        Args:
            maxWmLen: Maximum length of the watermark.
            vocabSize: Size of the vocabulary for watermark embedding. Should be 128 for 7-bit ASCII.
        """
        super().__init__()
        self.maxWmLen = maxWmLen
        self.vocabSize = vocabSize

    @staticmethod
    def ResBlock(inputs, filters):
        x = layers.Conv2D(filters, 3, padding='same')(inputs)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.Activation('relu')(x)
        return x

    @staticmethod
    def EncoderBlock(inputs, filters):
        x = BaseModel.ResBlock(inputs, filters)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        return x

    @staticmethod
    def DecoderBlock(inputs, skip, filters):
        x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(inputs)
        s = tf.image.resize(skip, tf.shape(x)[1:3]) # Resize skip connection
        x = layers.Concatenate()([x, s])
        x = BaseModel.ResBlock(x, filters)
        return x

    @staticmethod
    def SpatialAttentionBlock(inputs):
        x = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(inputs)
        return x

    @staticmethod
    def ContentAdaptiveStrengthBlock(inputs):
        x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(inputs)
        return x

class WatermarkEmbedderModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize encoder blocks filters and layers
        self.encoderFilters      = [ 64, 128, 256, 512 ]
        self.bottleneckFilters   = 1024
        self.decoderFilters      = [ 512, 256, 128, 64 ]
        self.finalConv           = keras.Sequential([ layers.Conv2D(3, 3, padding='same', activation='sigmoid') ])
        self.watermarkDense      = layers.Dense(1024, activation='relu')

        self.model = self.buildModel((None, None, 3), (self.maxWmLen))

    def buildModel(self, xShape, wmShape):
        x_in = layers.Input(shape=xShape)  # Dynamic input shape
        wm_in = layers.Input(shape=wmShape)
        x, watermark = x_in, wm_in

        # 1. Encode watermark into embedding
        wmEmbed = self.watermarkDense(tf.one_hot(tf.cast(watermark, tf.int32), depth=self.vocabSize))  # (B, maxWmLen, 1024)
        wmEmbed = tf.reduce_mean(wmEmbed, axis=1)                                   # (B, 1024)

        # 2. Encoder
        h = x
        skips = []
        for filters in self.encoderFilters:
            h = BaseModel.EncoderBlock(h, filters)
            skips.append(h)

        # 3. Bottleneck + attention + strength
        h  = BaseModel.ResBlock(h, self.bottleneckFilters)
        sa = BaseModel.SpatialAttentionBlock(h)
        cs = BaseModel.ContentAdaptiveStrengthBlock(h)

        # 4. Adjust latent + embed watermark
        h = h * (sa + 0.1) * (cs + 0.1)  # Adjust strength according to content
        wmMap = tf.reshape(wmEmbed, [-1, 1, 1, self.bottleneckFilters]) # (B, 1, 1, 1024)
        wmMap = tf.tile(wmMap, [1, tf.shape(h)[1], tf.shape(h)[2], 1]) # Broadcast watermark map
        h += wmMap

        # 5. Decoder
        for filters in self.decoderFilters:
            h = BaseModel.DecoderBlock(h, skips.pop(), filters)

        # 6. Final output
        out = self.finalConv(h) # (B,H',W',3), pixel ∈ [0,1]
        out = tf.image.resize(out, size=tf.shape(x)[1:3], method='bilinear') # Resize output to match input size

        return keras.Model(inputs=[x_in, wm_in], outputs=out, name='WatermarkEmbedderModel')

    def call(self, x, watermark):
        """
        Args:
            x: (B, H, W, 3) — ảnh gốc, đã chuẩn hóa [0,1].
            watermark: (B, maxWmLen) — chuỗi watermark, mỗi phần tử ∈ [0, vocabSize-1]
        Returns:
            out: (B, H, W, 3) — ảnh đã nhúng watermark.
        """
        return self.model([x, watermark])

class WatermarkExtractorModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize encoder blocks filters and layers
        self.encoderFilters     = [ 64, 128, 256, 512 ]
        self.bottleneckFilters  = 1024
        # Head
        self.pool   = layers.GlobalAveragePooling2D()
        self.dense  = layers.Dense(self.maxWmLen * self.vocabSize)

        self.model = self.buildModel((None, None, 3))

    def buildModel(self, xShape):
        x_in = layers.Input(shape=xShape)

        h = x_in
        for filters in self.encoderFilters:
            h = BaseModel.EncoderBlock(h, filters)

        # Bottleneck + Apply attention mechanisms (to focus on regions where watermark is strong)
        h  = BaseModel.ResBlock(h, self.bottleneckFilters)
        sa = BaseModel.SpatialAttentionBlock(h)    # (B,H',W',1)
        cs = BaseModel.ContentAdaptiveStrengthBlock(h)     # (B,H',W',1)
        h  = h * (sa + 0.1) * (cs + 0.1)

        # Global pooling and dense layer to predict watermark
        h = self.pool(h)  # (B,1024)
        h = self.dense(h)  # (B, maxWmLen * vocabSize)
        h = tf.reshape(h, [-1, self.maxWmLen])

        # Get outputs
        out = tf.nn.softmax(h, axis=-1) # (B, maxWmLen, vocabSize)
        return keras.Model(inputs=[x_in], outputs=out, name='WatermarkExtractorModel')

    def call(self, x):
        """
        Args:
            x: (B, H, W, 3) — watermarked image
        Returns:
            w_hat: (B, maxWmLen, vocabSize) — predicted one-hot probabilities
        """
        return self.model([x])

class SACAWNLoss(Loss):
    def __init__(self, impWeight=1.0, robWeight=2.0, extWeight=1.5):
        super().__init__()
        self.impWeight = impWeight
        self.robWeight = robWeight
        self.extWeight = extWeight
        self.bce = BinaryCrossentropy()

    def __call__(self, img_orig, img_watermarked, wm_true, wm_pred):
        # L_imperceptibility: 1-SSIM
        ssim = tf.image.ssim(img_orig, img_watermarked, max_val=1.0)
        L_imp = 1.0 - tf.reduce_mean(ssim)

        # L_robustness + L_extraction
        L_rob = self.bce(wm_true, wm_pred)
        L_ext = self.bce(wm_true, wm_pred)

        total_loss = (self.impWeight * L_imp +
                      self.robWeight * L_rob +
                      self.extWeight * L_ext)

        return total_loss, L_imp, L_rob, L_ext
