import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from m import WatermarkEmbedderModel, WatermarkExtractorModel, SACAWNLoss

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2 as Optimizer

from tqdm import tqdm

class Trainer:
    def __init__(self, embedder: WatermarkEmbedderModel, extractor: WatermarkExtractorModel, optimizer: Optimizer,
                 loss_fn, vocab_size=128, max_wm_len=256):
        self.embedder = embedder
        self.extractor = extractor
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.vocab_size = vocab_size
        self.max_wm_len = max_wm_len

    def generate_batch(self):
        H = tf.random.uniform([], minval=64, maxval=256, dtype=tf.int32)  # random height
        W = tf.random.uniform([], minval=64, maxval=256, dtype=tf.int32)  # random width

        imgs = tf.random.uniform((1, H, W, 3), 0, 1)
        wms  = tf.random.uniform((1, self.max_wm_len), minval=0, maxval=self.vocab_size, dtype=tf.int32)
        return imgs, wms

    @tf.function(reduce_retracing=True)
    def train_step(self, imgs, wms):
        with tf.GradientTape() as tape:
            # Embed
            watermarked_imgs = self.embedder(imgs, wms)

            # Extract
            wm_pred = self.extractor(watermarked_imgs)

            # Compute loss
            total_loss, L_imp, L_rob, L_ext = self.loss_fn(imgs, watermarked_imgs, wms, wm_pred)

        grads = tape.gradient(total_loss, self.embedder.trainable_variables + self.extractor.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.embedder.trainable_variables + self.extractor.trainable_variables))

        return total_loss, L_imp, L_rob, L_ext

    def train(self, epochs=10, steps_per_epoch=100):
        for epoch in range(1, epochs+1):
            print(f"\nEpoch {epoch}/{epochs}")
            epoch_loss = []
            for step in tqdm(range(steps_per_epoch), desc="Training", leave=False):
                imgs, wms = self.generate_batch()
                loss, L_imp, L_rob, L_ext = self.train_step(imgs, wms)
                epoch_loss.append(loss.numpy())

                if step % 10 == 0:
                    print(f"Step {step:03d} | Total: {loss.numpy():.4f} | L_imp: {L_imp.numpy():.4f} | L_rob: {L_rob.numpy():.4f} | L_ext: {L_ext.numpy():.4f}")

            print(f"Epoch {epoch} finished. Mean loss: {sum(epoch_loss)/len(epoch_loss):.4f}")
            # Save weights
            print("Saving weights...")
            self.embedder.save_weights("embedder_weights.h5")
            self.extractor.save_weights("extractor_weights.h5")

        # Save weights
        self.embedder.save_weights("embedder_weights.h5")
        self.extractor.save_weights("extractor_weights.h5")
############### 

VOCAB_SIZE = 128
MAX_WM_LEN = 256

embedder = WatermarkEmbedderModel(maxWmLen=MAX_WM_LEN, vocabSize=VOCAB_SIZE)
extractor = WatermarkExtractorModel(maxWmLen=MAX_WM_LEN, vocabSize=VOCAB_SIZE)

optimizer = Adam(1e-4)
loss_fn = SACAWNLoss(impWeight=1.0, robWeight=2.0, extWeight=1.5)
trainer = Trainer(embedder, extractor, optimizer, loss_fn, vocab_size=VOCAB_SIZE, max_wm_len=MAX_WM_LEN)

trainer.train(epochs=1, steps_per_epoch=1)