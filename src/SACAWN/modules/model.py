import tensorflow as tf
import keras
from keras import layers

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
            watermark: (B, L, S) — watermark text, L is the length of the watermark, S is the vocab size
        Returns:
            out: (B, H, W, 3) — image with embedded watermark.
        """
        image_input = layers.Input(shape=(None, None, 3), name="image_input")
        watermark_input = layers.Input(shape=(self.wmMaxLen, self.wmVocabSize), name="watermark_input")
        # inputs = layers.Input(shape=((None, None, None, 3), (None, self.wmMaxLen)))
        image, watermark = image_input, watermark_input

        # 1. Encode watermark into embedding
        wmEmbed = layers.Dense(1024, activation='relu')(watermark)  # (L, 1024)
        wmEmbed = layers.GlobalAveragePooling1D()(wmEmbed)          # (1024)

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

        # 3. Bottleneck + attention + strength & Adjust latent
        ex = BaseModel.ResBlock(ex, 1024)               # (H/32, W/32, 1024)
        sa = BaseModel.SpatialAttentionBlock(ex)        # (H/32, W/32, 1)
        cs = BaseModel.ContentAdaptiveStrengthBlock(ex) # (H/32, W/32, 1)
        ex = layers.Multiply()([ex, sa, cs])

        # 4. Embed watermark
        wmMap = layers.Reshape((1, 1, 1024))(wmEmbed)  # (B, 1, 1, 1024)
        mask = layers.Conv2D( # Create a broadcast mask with same HxW as `ex` but single channel
            filters=1, kernel_size=1, activation='linear', use_bias=False,
            kernel_initializer='ones', trainable=False
        )(ex * 0 + 1) # (B, H, W, 1)
        wmMapBroadcast = layers.Multiply()([wmMap, mask])  # (B, H, W, 1024)
        ex = layers.Add()([ex, wmMapBroadcast])

        # 5. Decoder
        ex = layers.Conv2DTranspose(512, (2, 2), strides=2, padding='same')(ex) # (H/16, W/16, 1024)
        ex = layers.Concatenate()([drops[3], ex])                               # Merge with drop features
        ex = BaseModel.ResBlock(ex, 512)                                        # (H/16, W/16, 512)

        ex = layers.Conv2DTranspose(256, (2, 2), strides=2, padding='same')(ex) # (H/8, W/8, 512)
        ex = layers.Concatenate()([drops[2], ex])                               # Merge with drop features
        ex = BaseModel.ResBlock(ex, 256)                                        # (H/8, W/8, 256)

        ex = layers.Conv2DTranspose(128, (2, 2), strides=2, padding='same')(ex) # (H/4, W/4, 256)
        ex = layers.Concatenate()([drops[1], ex])                               # Merge with drop features
        ex = BaseModel.ResBlock(ex, 128)                                        # (H/4, W/4, 128)

        ex = layers.Conv2DTranspose(64, (2, 2), strides=2, padding='same')(ex)  # (H/2, W/2, 128)
        ex = layers.Concatenate()([drops[0], ex])                               # Merge with drop features
        ex = BaseModel.ResBlock(ex, 64)                                         # (H/2, W/2, 64)

        # 6. Final output
        out = layers.Conv2D(3, 3, padding='same', activation='sigmoid')(ex) # (H',W',3), pixel ∈ [0,1]

        # Resize output to match input size
        out = layers.Lambda(
            lambda x: tf.image.resize(x[0], tf.shape(x[1])[1:3], method='bilinear')
        )([out, image])

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
        ex = BaseModel.ResBlock(ex, 1024)                           # Bottleneck step (H/32, W/32, 1024)

        # 2. Global pooling to create a fixed-size representation of the image
        # This resolves the `None` dimension issue.
        ex_pooled = layers.GlobalAveragePooling2D()(ex)     # (B, 1024)

        # 3. Create a sequence of length `wmMaxLen` from the pooled features
        seq = layers.RepeatVector(self.wmMaxLen)(ex_pooled) # (B, wmMaxLen, 1024)

        # 4. Apply Bidirectional LSTM for sequence modeling
        seq = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(seq)

        # 5. TimeDistributed Dense for vocab prediction
        out = layers.TimeDistributed(
            layers.Dense(self.wmVocabSize, activation='softmax')
        )(seq) # (B, L, S)

        model = keras.Model(inputs=[input], outputs=out, name='WatermarkExtractorModel')
        return model
