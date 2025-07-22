import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

from tensorflow import __version__; keras.__version__ = __version__

class BaseModel(keras.Model):
    def __init__(self, wmMaxLen=256, wmVocabSize=128):
        """
        Initialize the watermark embedder model.
        Args:
            wmMaxLen: Maximum length of the watermark.
            wmVocabSize: Size of the vocabulary for watermark embedding. Should be 128 for 7-bit ASCII.
        """
        super().__init__()
        self.wmMaxLen = wmMaxLen
        self.wmVocabSize = wmVocabSize

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

        self.model = self.buildModel((None, None, 3), (self.wmMaxLen))

    def buildModel(self, imgShape, wmShape):
        img_in  = layers.Input(shape=imgShape) # (B, H, W, 3)
        wm_in   = layers.Input(shape=wmShape) # (B, L)
        img, wm = img_in, wm_in

        # 1. Encode watermark into embedding
        wm = tf.cast(wm, dtype=tf.int32)
        wmEmbed = self.watermarkDense(tf.one_hot(wm, depth=self.wmVocabSize))  # (B, L, 1024)
        wmEmbed = tf.reduce_mean(wmEmbed, axis=1)                              # (B, 1024)

        # 2. Encoder
        ex = img
        skips = []
        for filters in self.encoderFilters:
            ex = BaseModel.EncoderBlock(ex, filters)
            skips.append(ex)

        # 3. Bottleneck + attention + strength
        ex = BaseModel.ResBlock(ex, self.bottleneckFilters)
        sa = BaseModel.SpatialAttentionBlock(ex)
        cs = BaseModel.ContentAdaptiveStrengthBlock(ex)

        # 4. Adjust latent + embed watermark
        ex = ex * (sa + 0.1) * (cs + 0.1)  # Adjust strength according to content
        wmMap = tf.reshape(wmEmbed, [-1, 1, 1, self.bottleneckFilters]) # (B, 1, 1, 1024)
        wmMap = tf.tile(wmMap, [1, tf.shape(ex)[1], tf.shape(ex)[2], 1]) # Broadcast watermark map
        ex += wmMap

        # 5. Decoder
        for filters in self.decoderFilters:
            ex = BaseModel.DecoderBlock(ex, skips.pop(), filters)

        # 6. Final output
        out = self.finalConv(ex) # (B,H',W',3), pixel ∈ [0,1]
        out = tf.image.resize(out, size=tf.shape(img_in)[1:3]) # Resize output to match input size

        return keras.Model(inputs=[img_in, wm_in], outputs=out, name='WatermarkEmbedderModel')

    def variables(self):
        """
        Get the variables of the model.
        """
        return self.model.trainable_variables \
            + self.trainable_variables \
            + self.watermarkDense.trainable_variables \
            + self.finalConv.trainable_variables

    def call(self, image, watermark):
        """
        Args:
            image: (B, H, W, 3) — raw image, pixel ∈ [0,1]
            watermark: (B, L, 1) — watermark text, L is the length of the watermark.
        Returns:
            out: (B, H, W, 3) — image with embedded watermark.
        """
        return self.model([image, watermark])

class WatermarkExtractorModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize encoder blocks filters and layers
        self.encoderFilters     = [ 64, 128, 256, 512 ]
        self.bottleneckFilters  = 1024
        # Head
        self.pool   = layers.GlobalAveragePooling2D()
        self.dense  = keras.Sequential([
            layers.Dense(self.wmMaxLen * self.wmVocabSize),
            layers.Activation('relu'),
            layers.Dense(self.wmMaxLen),
        ])

        self.model = self.buildModel((None, None, 3))

    def buildModel(self, imgShape):
        img_in = layers.Input(shape=imgShape)

        ex = img_in
        for filters in self.encoderFilters:
            ex = BaseModel.EncoderBlock(ex, filters)

        # Bottleneck + Apply attention mechanisms (to focus on regions where watermark is strong)
        ex = BaseModel.ResBlock(ex, self.bottleneckFilters)
        sa = BaseModel.SpatialAttentionBlock(ex)            # (B,H',W',1)
        cs = BaseModel.ContentAdaptiveStrengthBlock(ex)     # (B,H',W',1)
        ex = ex * (sa + 0.1) * (cs + 0.1)

        # Global pooling and dense layer to predict watermark
        ex  = self.pool(ex)   # (B, 1024)
        out = self.dense(ex)  # (B, wmMaxLen)

        return keras.Model(inputs=[img_in], outputs=out, name='WatermarkExtractorModel')

    def variables(self):
        """
        Get the variables of the model.
        """
        return self.model.trainable_variables \
            + self.trainable_variables \
            + self.pool.trainable_variables \
            + self.dense.trainable_variables

    def call(self, img):
        """
        Args:
            img: (B, H, W, 3) — watermarked image
        Returns:
            out: (B, wmMaxLen) — predicted one-hot probabilities
        """
        return self.model([img]) # (B, wmMaxLen)
