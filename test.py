import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import matplotlib.pyplot as plt
from m import WatermarkEmbedderModel, WatermarkExtractorModel, SACAWNLoss

# ======== Config ========
VOCAB_SIZE = 128
MAX_WM_LEN = 256
IMG_SIZE = (128,128)

# ======== Load models ========
embedder = WatermarkEmbedderModel(maxWmLen=MAX_WM_LEN, vocabSize=VOCAB_SIZE)
extractor = WatermarkExtractorModel(maxWmLen=MAX_WM_LEN, vocabSize=VOCAB_SIZE)

embedder.load_weights("embedder_weights.h5")
extractor.load_weights("extractor_weights.h5")

loss_fn = SACAWNLoss(impWeight=1.0, robWeight=2.0, extWeight=1.5)

# ======== Generate dummy test data ========
BATCH_SIZE = 1
H, W = 128, 128
imgs = tf.random.uniform((BATCH_SIZE, H, W, 3), 0, 1)
wms  = tf.random.uniform((BATCH_SIZE, MAX_WM_LEN), minval=0, maxval=VOCAB_SIZE, dtype=tf.int32)

# ======== Run models ========
watermarked_imgs = embedder(imgs, wms)
wm_pred = extractor(watermarked_imgs)

loss, L_imp, L_rob, L_ext = loss_fn(imgs, watermarked_imgs, wms, wm_pred)

print(f"Test Loss: {loss.numpy():.4f}")
print(f"  L_imperceptibility: {L_imp.numpy():.4f}")
print(f"  L_robustness:       {L_rob.numpy():.4f}")
print(f"  L_extraction:       {L_ext.numpy():.4f}")

# ======== Visualization ========
def show_img(img, title):
    img = tf.squeeze(img)  # remove batch
    plt.imshow(img.numpy())
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
show_img(imgs[0], "Original Image")
plt.subplot(1,2,2)
show_img(watermarked_imgs[0], "Watermarked Image")
plt.show()

# ======== Show predicted watermark (optional) ========
wm_pred_indices = tf.argmax(wm_pred, axis=-1).numpy()
print("Predicted watermark indices (first 20):", wm_pred_indices[0][:20])
print("Ground truth watermark indices (first 20):", wms.numpy()[0][:20])
