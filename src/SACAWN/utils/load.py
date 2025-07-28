import tensorflow as tf

def loadImageTensor(path: str) -> tf.Tensor: # (H, W, 3)
    """
    Load an image from the given path and convert it to a tf.Tensor.

    Args:
        path (str): Path to the image file.
    Returns:
        tf.Tensor: Image tensor with pixel values in the range [0, 1].
    """
    imgRaw = tf.io.read_file(path)
    if path.lower().endswith(".png"):
        img = tf.image.decode_png(imgRaw, channels=3)
    else:
        img = tf.image.decode_jpeg(imgRaw, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32) # [0,1]
    return img

def loadStringTensor(text: str, size: int, padChar: bytes = b'\0') -> tf.Tensor: # (size)
    """
    Convert a string to a tf.Tensor.

    Args:
        text (str): Input string.
        size (int): Desired size of the output tensor.
    Returns:
        tf.Tensor: Tensor representation of the string.
    """
    # Encode text as bytes
    textBytes = text.encode("utf-8")
    # Truncate if longer, or pad if shorter
    if len(textBytes) > size:
        textBytes = textBytes[:size]
    else:
        textBytes += padChar * (size - len(textBytes))
    return tf.reshape(tf.constant([b for b in textBytes], dtype=tf.float32), (size,))
