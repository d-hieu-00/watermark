import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import matplotlib.pyplot as plt
from m import WatermarkEmbedderModel, WatermarkExtractorModel, SACAWNLoss
from PIL import Image
import numpy as np

# ======== Config ========
VOCAB_SIZE = 128
MAX_WM_LEN = 256
IMG_PATH = "data/a.jpg"      # 📷 đường dẫn ảnh thật
RESULT_DIR = "results"                   # 📁 thư mục lưu kết quả

os.makedirs(RESULT_DIR, exist_ok=True)

# ======== Load models ========
embedder = WatermarkEmbedderModel(maxWmLen=MAX_WM_LEN, vocabSize=VOCAB_SIZE)
extractor = WatermarkExtractorModel(maxWmLen=MAX_WM_LEN, vocabSize=VOCAB_SIZE)

embedder.load_weights("embedder_weights.h5")
extractor.load_weights("extractor_weights.h5")

loss_fn = SACAWNLoss(impWeight=1.0, robWeight=2.0, extWeight=1.5)

# ======== Load and preprocess image ========
def load_image(path, max_size=256):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(max_size/w, max_size/h, 1.0)
    img = img.resize((int(w*scale), int(h*scale)))
    img = np.array(img) / 255.0
    return img

img_np = load_image(IMG_PATH)
H, W = img_np.shape[:2]
imgs = tf.convert_to_tensor(img_np[np.newaxis, ...], dtype=tf.float32)

# ======== Generate watermark ========
wms = tf.random.uniform((1, MAX_WM_LEN), minval=0, maxval=VOCAB_SIZE, dtype=tf.int32) # random watermark sequence by char in ascii range [0, VOCAB_SIZE-1]

# ======== Run models ========
watermarked_imgs = embedder(imgs, wms)
wm_pred = extractor(watermarked_imgs)

loss, L_imp, L_rob, L_ext = loss_fn(imgs, watermarked_imgs, wms, wm_pred)

print(f"Test Loss: {loss.numpy():.4f}")
print(f"  L_imperceptibility: {L_imp.numpy():.4f}")
print(f"  L_robustness:       {L_rob.numpy():.4f}")
print(f"  L_extraction:       {L_ext.numpy():.4f}")

# ======== Save results ========
def save_image(img_tensor, path):
    img = tf.squeeze(img_tensor)  # (H,W,3)
    img = tf.clip_by_value(img, 0, 1).numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)

save_image(imgs, os.path.join(RESULT_DIR, "original.png"))
save_image(watermarked_imgs, os.path.join(RESULT_DIR, "watermarked.png"))

# ======== Show results ========
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(tf.squeeze(imgs).numpy())
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(tf.squeeze(watermarked_imgs).numpy())
plt.title("Watermarked Image")
plt.axis('off')
plt.savefig(os.path.join(RESULT_DIR, "comparison.png"))
plt.show()

# ======== Show predicted watermark ========
wm_pred_indices = tf.argmax(wm_pred, axis=-1).numpy()
gt_wm_indices  = wms.numpy()

with open(os.path.join(RESULT_DIR, "watermark.txt"), "w") as f:
    f.write("Ground truth watermark indices (first 20):\n")
    f.write(str(gt_wm_indices[0][:20]) + "\n")
    f.write("Predicted watermark indices (first 20):\n")
    f.write(str(wm_pred_indices[0][:20]) + "\n")

print(f"✅ Results saved in: {RESULT_DIR}")