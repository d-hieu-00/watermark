import os
import math

class BaseLoader:
    def __init__(self, batchSize=1):
        self.__batchSize = batchSize

    @property
    def batchSize(self):
        return self.__batchSize

    @batchSize.setter
    def batchSize(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Batch size must be a positive integer.")
        self.__batchSize = value

    @property
    def totalBatches(self):
        """
        Calculate the total number of batches based on the batch size.
        """
        data_len = self.len()
        if data_len == 0:
            return 0
        return math.ceil(data_len / self.__batchSize)

    def len(self):
        """
        Get the length of the dataset. Should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def next(self):
        """
        Get the next item. Should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def nextBatch(self):
        """
        Get the next batch of items.
        Returns:
            tuple: (batch_data, reached_end_of_epoch)
                   batch_data will be a list of items, or a tuple of tf.Tensors if processed.
                   reached_end_of_epoch is a boolean indicating if the loader reset.
        """
        batch = []
        reachedEnd = False
        for _ in range(self.__batchSize):
            item = self.next()
            if item is None:
                reachedEnd = True
                break # Stop collecting if we hit the end of an epoch
            batch.append(item)
        return batch, reachedEnd

# Read watermarks from a file
class WatermarkLoader(BaseLoader):
    def __init__(self, watermarkFile, batchSize=1):
        super().__init__(batchSize=batchSize)
        self.__watermarks    = []
        self.__idx           = 0
        # Expect the total number of watermarks to be small
        # Then we can load them all into memory
        with open(watermarkFile, 'r', encoding='latin1') as f:
            for line in f:
                wm = line.strip()
                self.__watermarks.append(wm)

    # Length of the watermarks
    def len(self):
        return len(self.__watermarks)

    # Get the next watermark string
    def next(self):
        if self.__idx >= self.len():
            # Reset the index if we reach the end of the watermarks list
            self.__idx = 0
            return None # Signal end of watermark cycle
        watermark = self.__watermarks[self.__idx]
        self.__idx += 1
        return watermark

# Load images from a directory
class ImageLoader(BaseLoader):
    def __init__(self, imageDir, batchSize=1):
        super().__init__(batchSize=batchSize)
        self.__imageFiles   = []
        self.__idx          = 0

        # Only load image file names that end with .png, .jpg, or .jpeg
        # Then we can load them all into memory --> Doesn't take much memory
        for root, _, files in os.walk(imageDir):
            for filename in files:
                if isinstance(filename, str) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.__imageFiles.append(os.path.join(root, filename))

    # Length of the image files
    def len(self):
        return len(self.__imageFiles)

    # Get the next image file path
    def next(self):
        if self.__idx >= self.len():
            # Reset the index if we reach the end of the image files list
            self.__idx = 0
            return None # Signal end of epoch for images
        image_path = self.__imageFiles[self.__idx]
        self.__idx += 1
        return image_path


# Train and validation data loader
class DataLoader(BaseLoader):
    def __init__(self, imageDir, watermarkFile, batchSize=1):
        super().__init__(batchSize=batchSize)
        self.imageLoader = ImageLoader(imageDir, batchSize=1)
        self.watermarkLoader = WatermarkLoader(watermarkFile, batchSize=1)

    def len(self):
        # Total number of samples is determined by the number of images
        return self.imageLoader.len()

    def next(self):
        """
        Return the next (image, watermark) pair.
        Images move forward one by one.
        Watermarks cycle when exhausted.
        """
        image = self.imageLoader.next()
        if image is None:
            # end of epoch for images
            return None

        watermark = self.watermarkLoader.next()
        # if watermark None, it just means the watermarkLoader reset itself
        # and its next() already restarted from the beginning
        if watermark is None:
            watermark = self.watermarkLoader.next()

        return (image, watermark)
