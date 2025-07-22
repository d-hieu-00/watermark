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
    def ResBlock(filters):
        return keras.Sequential([
            layers.Conv2D(filters, 3, padding='same'),
            layers.Activation('relu'),
            layers.Conv2D(filters, 3, padding='same'),
            layers.Activation('relu'),
        ])

    # @staticmethod
    # def EncoderBlock(filters):
    #     return keras.Sequential([
    #         layers.Conv2D(filters, 3, padding='same'),
    #         layers.Activation('relu'),
    #         layers.Conv2D(filters, 3, padding='same'),
    #         layers.Activation('relu'),
    #     ])
    #     x = BaseModel.ResBlock(filters)
    #     x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    #     return x

    # @staticmethod
    # def DecoderBlock(inputs, skip, filters):
    #     x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(inputs)
    #     s = tf.image.resize(skip, tf.shape(x)[1:3]) # Resize skip connection
    #     x = layers.Concatenate()([x, s])
    #     x = BaseModel.ResBlock(x, filters)
    #     return x

    @staticmethod
    def SpatialAttentionBlock():
        return keras.Sequential([
            layers.Conv2D(1, 7, padding='same', activation='sigmoid'),
        ])

    @staticmethod
    def ContentAdaptiveStrengthBlock():
        return keras.Sequential([
            layers.Conv2D(1, 3, padding='same', activation='sigmoid'),
        ])

class WatermarkEmbedderModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize encoder blocks filters and layers
        self.encoder1       = BaseModel.ResBlock(64)
        self.encoderDown1   = layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        self.encoder2       = BaseModel.ResBlock(128)
        self.encoderDown2   = layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        self.encoder3       = BaseModel.ResBlock(256)
        self.encoderDown3   = layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        self.encoder4       = BaseModel.ResBlock(512)
        self.encoderDown4   = layers.MaxPooling2D(pool_size=(2, 2), strides=2)

        self.bottleneckFilters          = 1024
        self.bottleneck                 = BaseModel.ResBlock(self.bottleneckFilters)
        self.spatialAttention           = BaseModel.SpatialAttentionBlock()
        self.contentAdaptiveStrength    = BaseModel.ContentAdaptiveStrengthBlock()

        self.decoderUp1     = layers.UpSampling2D(size=(2, 2))
        self.decoder1       = BaseModel.ResBlock(512)
        self.decoderUp2     = layers.UpSampling2D(size=(2, 2))
        self.decoder2       = BaseModel.ResBlock(256)
        self.decoderUp3     = layers.UpSampling2D(size=(2, 2))
        self.decoder3       = BaseModel.ResBlock(128)
        self.decoderUp4     = layers.UpSampling2D(size=(2, 2))
        self.decoder4       = BaseModel.ResBlock(64)

        self.finalConv           = keras.Sequential([ layers.Conv2D(3, 3, padding='same', activation='sigmoid') ])
        self.watermarkDense      = layers.Dense(self.bottleneckFilters, activation='relu')

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
        ex = self.encoder1(ex); skips.append(ex)    # (B, H/2, W/2, 64)
        ex = self.encoderDown1(ex)                  # (B, H/4, W/4, 64)
        ex = self.encoder2(ex); skips.append(ex)    # (B, H/4, W/4, 128)
        ex = self.encoderDown2(ex)                  # (B, H/8, W/8, 128)
        ex = self.encoder3(ex); skips.append(ex)    # (B, H/8, W/8, 256)
        ex = self.encoderDown3(ex)                  # (B, H/16, W/16, 256)
        ex = self.encoder4(ex); skips.append(ex)    # (B, H/16, W/16, 512)
        ex = self.encoderDown4(ex)                  # (B, H/32, W/32, 512)

        # 3. Bottleneck + attention + strength
        ex = self.bottleneck(ex)              # (B, H/32, W/32, 1024)
        sa = self.spatialAttention(ex)        # (B, H/32, W/32, 1)
        cs = self.contentAdaptiveStrength(ex) # (B, H/32, W/32, 1)

        # 4. Adjust latent + embed watermark
        ex = ex * (sa + 0.1) * (cs + 0.1)  # Adjust strength according to content
        wmMap = tf.reshape(wmEmbed, [-1, 1, 1, self.bottleneckFilters])  # (B, 1, 1, 1024)
        wmMap = tf.tile(wmMap, [1, tf.shape(ex)[1], tf.shape(ex)[2], 1]) # Broadcast watermark map
        ex += wmMap

        # 5. Decoder
        ex = self.decoderUp1(ex) # (B, H/16, W/16, 1024)
        sk = tf.image.resize(skips[3], tf.shape(ex)[1:3]) # Resize skip connection
        ex = layers.Concatenate()([ex, sk]) # Skip connection from encoder4
        ex = self.decoder1(ex) # (B, H/16, W/16, 512)

        ex = self.decoderUp2(ex) # (B, H/8, W/8, 512)
        ex = layers.Concatenate()([ex, skips[3]]) # Skip connection from encoder4
        sk = tf.image.resize(skips[2], tf.shape(ex)[1:3]) # Resize skip connection
        ex = layers.Concatenate()([ex, sk]) # Skip connection from encoder4
        ex = self.decoder2(ex) # (B, H/8, W/8, 256)

        ex = self.decoderUp3(ex) # (B, H/4, W/4, 256)
        sk = tf.image.resize(skips[1], tf.shape(ex)[1:3]) # Resize skip connection
        ex = layers.Concatenate()([ex, sk]) # Skip connection from encoder3
        ex = self.decoder3(ex) # (B, H/4, W/4, 128)

        ex = self.decoderUp4(ex) # (B, H/2, W/2, 128)
        sk = tf.image.resize(skips[0], tf.shape(ex)[1:3]) # Resize skip connection
        ex = layers.Concatenate()([ex, sk]) # Skip connection from encoder2
        ex = self.decoder4(ex) # (B, H/2, W/2, 64)

        # 6. Final output
        out = self.finalConv(ex) # (B,H',W',3), pixel ∈ [0,1]
        out = tf.image.resize(out, size=tf.shape(img_in)[1:3]) # Resize output to match input size

        return keras.Model(inputs=[img_in, wm_in], outputs=out, name='WatermarkEmbedderModel')

    def variables(self):
        """
        Get the variables of the model.
        """
        return self.trainable_variables
            # + self.model.trainable_variables \
            # + self.watermarkDense.trainable_variables \
            # + self.finalConv.trainable_variables

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
        self.encoder1       = BaseModel.ResBlock(64)
        self.encoderDown1   = layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        self.encoder2       = BaseModel.ResBlock(128)
        self.encoderDown2   = layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        self.encoder3       = BaseModel.ResBlock(256)
        self.encoderDown3   = layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        self.encoder4       = BaseModel.ResBlock(512)
        self.encoderDown4   = layers.MaxPooling2D(pool_size=(2, 2), strides=2)

        self.bottleneckFilters          = 1024
        self.bottleneck                 = BaseModel.ResBlock(self.bottleneckFilters)
        self.spatialAttention           = BaseModel.SpatialAttentionBlock()
        self.contentAdaptiveStrength    = BaseModel.ContentAdaptiveStrengthBlock()

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

        # 1. Encoder
        ex = img_in
        ex = self.encoder1(ex);     # (B, H/2, W/2, 64)
        ex = self.encoderDown1(ex)  # (B, H/4, W/4, 64)
        ex = self.encoder2(ex);     # (B, H/4, W/4, 128)
        ex = self.encoderDown2(ex)  # (B, H/8, W/8, 128)
        ex = self.encoder3(ex);     # (B, H/8, W/8, 256)
        ex = self.encoderDown3(ex)  # (B, H/16, W/16, 256)
        ex = self.encoder4(ex);     # (B, H/16, W/16, 512)
        ex = self.encoderDown4(ex)  # (B, H/32, W/32, 512)

        # 2. Bottleneck + attention + strength
        ex = self.bottleneck(ex)              # (B, H/32, W/32, 1024)
        sa = self.spatialAttention(ex)        # (B, H/32, W/32, 1)
        cs = self.contentAdaptiveStrength(ex) # (B, H/32, W/32, 1)

        # 3. Adjust latent
        ex = ex * (sa + 0.1) * (cs + 0.1)  # Adjust strength according to content

        # 4. Global pooling and dense layer to predict watermark
        ex  = self.pool(ex)   # (B, 1024)
        out = self.dense(ex)  # (B, wmMaxLen)

        return keras.Model(inputs=[img_in], outputs=out, name='WatermarkExtractorModel')

    def variables(self):
        """
        Get the variables of the model.
        """
        return self.model.trainable_variables

    def call(self, img):
        """
        Args:
            img: (B, H, W, 3) — watermarked image
        Returns:
            out: (B, wmMaxLen) — predicted one-hot probabilities
        """
        return self.model([img]) # (B, wmMaxLen)
