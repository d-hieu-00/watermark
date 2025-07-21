import tensorflow as tf

VOCAB_SIZE = 128
MAX_WM_LEN = 256

# Load models
embedder = tf.keras.models.load_model("embedder_full_model.h5")
extractor = tf.keras.models.load_model("extractor_full_model.h5")

# Load an example image
img_path = "example.jpg"   # 📝 Thay bằng ảnh thật của bạn
img = tf.io.read_file(img_path)
img = tf.image.decode_image(img, channels=3)
img = tf.image.convert_image_dtype(img, tf.float32)
img = tf.image.resize(img, (128, 128))
img = tf.expand_dims(img, axis=0)  # (1, H, W, 3)

# Dummy watermark
wm = tf.random.uniform((1, MAX_WM_LEN), minval=0, maxval=VOCAB_SIZE, dtype=tf.int32)

# Run embedder
watermarked = embedder([img, wm])

# Save watermarked image
watermarked_img = tf.squeeze(watermarked, axis=0)
watermarked_img = tf.clip_by_value(watermarked_img, 0.0, 1.0)
watermarked_img = tf.image.convert_image_dtype(watermarked_img, tf.uint8)
tf.keras.utils.save_img("watermarked_output.png", watermarked_img)
print("✅ Watermarked image saved to watermarked_output.png")

# Run extractor
predicted_wm = extractor(watermarked)
print("🔷 Predicted watermark shape:", predicted_wm.shape)
