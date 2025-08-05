import tensorflow as tf
import keras

from utils.load import loadImageTensor, loadStringTensor

MAX_WM_LEN = 256
VOCAB_SIZE = 128

def prepareInput(imageFile, watermarkText):
    imgTensor = loadImageTensor(imageFile)
    wmTensor = loadStringTensor(watermarkText, MAX_WM_LEN, VOCAB_SIZE)
    return (imgTensor, wmTensor)

def saveImage(tensor, filename):
    # Scale to [0, 255] and convert to uint8
    tensor = tf.image.convert_image_dtype(tensor, dtype=tf.uint8, saturate=True)
    # Encode as PNG
    png = tf.image.encode_png(tensor)
    # Write to file
    tf.io.write_file(filename, png)

def run(imageFile, watermarkText, embedder, extractor):
    imgTensor, wmTensor = prepareInput(imageFile, watermarkText)
    # Add batch dimension
    imgTensor = tf.expand_dims(imgTensor, axis=0)
    wmTensor = tf.expand_dims(wmTensor, axis=0)
    # Run embedder
    imgWm = embedder((imgTensor, wmTensor))
    # Run extractor
    if extractor is None:
        return imgWm, None
    extractWm = extractor(imgWm)
    # return the watermarked image and extracted watermark
    return imgWm, extractWm, wmTensor

# Load models
embedder = keras.models.load_model("./src/SACAWN/_result/20250730-191606/embedder.h5", compile=False)
extractor = keras.models.load_model("./src/SACAWN/_result/20250730-191606/extractor.h5", compile=False)

# Example usage
(imgWm, extractWm, embeddedWm) = run("./src/SACAWN/test.jpg", "Sample Watermark", embedder, extractor)

# Save watermarked image
saveImage(imgWm[0], "watermarked_output.png")
print("Watermarked image saved to watermarked_output.png")

# Run extractor
# Convert (256) to string
# extractWm = tf.squeeze(extractWm, axis=0)  # Remove batch

print("Extracted watermark :", tf.argmax(tf.squeeze(extractWm), axis=-1))
print("Embedded watermark :", tf.argmax(tf.squeeze(embeddedWm), axis=-1))
