import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

from tensorflow import __version__; keras.__version__ = __version__

class BaseModel:
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
    def ResBlock(input, numFilters):
        x = layers.Conv2D(numFilters, 3, padding='same')(input)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(numFilters, 3, padding='same')(x)
        x = layers.Activation('relu')(x)
        return x

    @staticmethod
    def SpatialAttentionBlock(input):
        return layers.Conv2D(1, 7, padding='same', activation='sigmoid')(input)

    @staticmethod
    def ContentAdaptiveStrengthBlock(input):
        return layers.Conv2D(1, 3, padding='same', activation='sigmoid')(input)

class WatermarkEmbedderModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self):
        """
        Args:
            xInput: tuple of (image, watermark)
            image: (B, H, W, 3) — raw image, pixel ∈ [0,1]
            watermark: (B, L) — watermark text, L is the length of the watermark.
        Returns:
            out: (B, H, W, 3) — image with embedded watermark.
        """
        image_input = layers.Input(shape=(None, None, 3), name="image_input")
        watermark_input = layers.Input(shape=(self.wmMaxLen,), name="watermark_input")
        # inputs = layers.Input(shape=((None, None, None, 3), (None, self.wmMaxLen)))
        image, watermark = image_input, watermark_input

        # 1. Encode watermark into embedding
        wm = tf.cast(watermark, dtype=tf.int32)
        wm = tf.one_hot(wm, depth=self.wmVocabSize)
        wmEmbed = layers.Dense(1024, activation='relu')(wm) # (L, 1024)
        wmEmbed = tf.reduce_mean(wmEmbed, axis=1)           # (1024)

        # 2. Encoder
        ex = image
        drops = []
        ex = BaseModel.ResBlock(ex, 64); drops.append(layers.Dropout(0.5)(ex))  # (H/2, W/2, 64)
        ex = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(ex)               # (H/4, W/4, 64)
        ex = BaseModel.ResBlock(ex, 128); drops.append(layers.Dropout(0.5)(ex)) # (H/4, W/4, 128)
        ex = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(ex)               # (H/8, W/8, 128)
        ex = BaseModel.ResBlock(ex, 256); drops.append(layers.Dropout(0.5)(ex)) # (H/8, W/8, 256)
        ex = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(ex)               # (H/16, W/16, 256)
        ex = BaseModel.ResBlock(ex, 512); drops.append(layers.Dropout(0.5)(ex)) # (H/16, W/16, 512)
        ex = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(ex)               # (H/32, W/32, 512)

        # 3. Bottleneck + attention + strength
        ex = BaseModel.ResBlock(ex, 1024)               # (H/32, W/32, 1024)
        sa = BaseModel.SpatialAttentionBlock(ex)        # (H/32, W/32, 1)
        cs = BaseModel.ContentAdaptiveStrengthBlock(ex) # (H/32, W/32, 1)

        # 4. Adjust latent + embed watermark
        ex = ex * (sa + 0.1) * (cs + 0.1)  # Adjust strength according to content
        wmMap = tf.reshape(wmEmbed, [-1, 1, 1, 1024])  # (1, 1, 1024)
        wmMap = tf.tile(wmMap, [1, tf.shape(ex)[1], tf.shape(ex)[2], 1]) # Broadcast watermark map
        ex += wmMap

        # 5. Decoder
        ex = layers.Conv2DTranspose(512, (2, 2), strides=2, padding='same')(ex) # (H/16, W/16, 1024)
        dr = tf.image.resize(drops[3], tf.shape(ex)[1:3])                       # Resize drop features
        ex = layers.Concatenate()([dr, ex])                                     # Merge with drop features
        ex = BaseModel.ResBlock(ex, 512)                                        # (H/16, W/16, 512)

        ex = layers.Conv2DTranspose(256, (2, 2), strides=2, padding='same')(ex) # (H/8, W/8, 512)
        dr = tf.image.resize(drops[2], tf.shape(ex)[1:3])                       # Resize drop features
        ex = layers.Concatenate()([dr, ex])                                     # Merge with drop features
        ex = BaseModel.ResBlock(ex, 256)                                        # (H/8, W/8, 256)

        ex = layers.Conv2DTranspose(128, (2, 2), strides=2, padding='same')(ex) # (H/4, W/4, 256)
        dr = tf.image.resize(drops[1], tf.shape(ex)[1:3])                       # Resize drop features
        ex = layers.Concatenate()([dr, ex])                                     # Merge with drop features
        ex = BaseModel.ResBlock(ex, 128)                                        # (H/4, W/4, 128)

        ex = layers.Conv2DTranspose(64, (2, 2), strides=2, padding='same')(ex)  # (H/2, W/2, 128)
        dr = tf.image.resize(drops[0], tf.shape(ex)[1:3])                       # Resize drop features
        ex = layers.Concatenate()([dr, ex])                                     # Merge with drop features
        ex = BaseModel.ResBlock(ex, 64)                                         # (H/2, W/2, 64)

        # 6. Final output
        out = layers.Conv2D(3, 3, padding='same', activation='sigmoid')(ex) # (H',W',3), pixel ∈ [0,1]
        out = tf.image.resize(out, size=tf.shape(image)[1:3]) # Resize output to match input size

        model = keras.Model(inputs=[image_input, watermark_input], outputs=out, name='WatermarkEmbedderModel')
        return model

class WatermarkExtractorModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self):
        """
        Args:
            img: (H, W, 3) — watermarked image
        Returns:
            out: (wmMaxLen) — predicted one-hot probabilities
        """

        # 1. Encoder
        input = layers.Input(shape=(None, None, 3))
        ex = input
        ex = BaseModel.ResBlock(ex, 64);                            # (H/2, W/2, 64)
        ex = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(ex)   # (H/4, W/4, 64)
        ex = BaseModel.ResBlock(ex, 128);                           # (H/4, W/4, 128)
        ex = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(ex)   # (H/8, W/8, 128)
        ex = BaseModel.ResBlock(ex, 256);                           # (H/8, W/8, 256)
        ex = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(ex)   # (H/16, W/16, 256)
        ex = BaseModel.ResBlock(ex, 512);                           # (H/16, W/16, 512)
        ex = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(ex)   # (H/32, W/32, 512)

        # 2. Bottleneck
        ex = BaseModel.ResBlock(ex, 1024)               # (H/32, W/32, 1024)

        # 4. Global pooling and dense layer to predict watermark
        ex  = layers.GlobalAveragePooling2D()(ex) # (1024)
        ex = layers.Dense(self.wmMaxLen * self.wmVocabSize, activation='relu')(ex)

        # 5. Reshape the output to (BatchSize, wmMaxLen, wmVocabSize) & Apply softmax activation to get probabilities for each character at each position
        out = layers.Reshape((self.wmMaxLen, self.wmVocabSize))(ex)
        out = layers.Activation('softmax')(out) # (B, wmMaxLen, wmVocabSize)

        model = keras.Model(inputs=[input], outputs=out, name='WatermarkExtractorModel')
        return model
