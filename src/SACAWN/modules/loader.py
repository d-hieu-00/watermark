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
        self.__batchSize = value

    @property
    def totalBatches(self):
        """
        Calculate the total number of batches based on the batch size.
        """
        return math.ceil(self.len() / self.__batchSize)

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
        Get the next batch of items. Should be implemented in subclasses.
        """
        batch = []
        reachedEnd = False
        for _ in range(self.__batchSize):
            item = self.next()
            if item is None:
                reachedEnd = True
                break
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
                if len(wm) < 256:
                    wm += ' ' * (256 - len(wm))  # Pad to 256 characters
                self.__watermarks.append(wm)

    # Length of the watermarks
    def len(self):
        return len(self.__watermarks)

    # Get the next watermark
    def next(self):
        if self.__idx >= self.len():
            # Reset the index if we reach the end
            self.__idx = 0
            return None
        # Get the watermark at the current index
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
                if isinstance(filename, str) and filename.endswith(('.png', '.jpg', '.jpeg')):
                    self.__imageFiles.append(os.path.join(root, filename))

    # Length of the watermarks
    def len(self):
        return len(self.__imageFiles)

    # Get the current index
    def next(self):
        if self.__idx >= self.len():
            # Reset the index if we reach the end
            self.__idx = 0
            return None
        image = self.__imageFiles[self.__idx]
        self.__idx += 1
        return image


# Train and validation data loader
class DataLoader(BaseLoader):
    def __init__(self, imageDir, watermarkFile, batchSize=1):
        super().__init__(batchSize=batchSize)
        self.imageLoader = ImageLoader(imageDir, batchSize=1)
        self.watermarkLoader = WatermarkLoader(watermarkFile, batchSize=1)
    
    def len(self):
        # total number of samples = number of images
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
