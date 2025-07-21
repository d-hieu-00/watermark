
# External
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2 as Optimizer

from tqdm import tqdm # Progress bar for training loop

# Internal
import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parent.parent))
from modules.loader import DataLoader

logger = logging.getLogger(__name__)


def string_to_tensor(s: str, max_len=256, vocab_size=128):
    """
    Convert string s to a tf.Tensor of shape (256,) padded/truncated to length 256.
    Characters are encoded as ASCII (or clipped to vocab_size).

    Args:
        s: string to encode
        max_len: length to pad/truncate
        vocab_size: clip values to this range

    Returns:
        tf.Tensor of shape (256,) and dtype int32
    """
    # Encode string to bytes → int list
    ids = [ord(c) for c in s]

    # Truncate
    ids = ids[:max_len]

    # Pad if too short
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))

    # Clip values to vocab size
    ids = np.clip(ids, 0, vocab_size-1)

    # Convert to tf.Tensor
    return tf.constant(ids, dtype=tf.int32)

class Trainer:

    @staticmethod
    def LossesHeaders():
        tf.random.uniform((1,))  # Ensure TensorFlow is initialized
        return ["timestamp", "epoch", "step", "total_loss", "imperceptibility_loss", "robustness_loss", "extraction_loss"]

    @staticmethod
    def TrainLossesFilename():
        return "train_loss_history.csv"

    @staticmethod
    def ValidationLossesFilename():
        return "val_loss_history.csv"

    def __init__(self, embedder: keras.Model, extractor: keras.Model, optimizer: Optimizer, trainLoader: DataLoader, valLoader: DataLoader, lossFn, outputDir):
        self.embedder       = embedder
        self.extractor      = extractor
        self.optimizer      = optimizer
        self.trainLoader    = trainLoader
        self.valLoader      = valLoader
        self.lossFn         = lossFn
        self.outputDir      = outputDir

        # Check if loss history files exist, if not create it
        self._checkLossHistoryFile(Trainer.TrainLossesFilename())
        self._checkLossHistoryFile(Trainer.ValidationLossesFilename())

    def _checkLossHistoryFile(self, filename):
        """
        Check if the loss history file exists, if not create it.
        """
        try:
            with open(f"{self.outputDir}/{filename}", 'x') as f:
                f.write(",".join(Trainer.LossesHeaders()) + "\n")
        except FileExistsError:
            pass

    def _prepareBatch(self, loader: DataLoader):
        """
        Prepare a batch of images and watermarks from the loader.
        Args:
            dataloader: an instance of your DataLoader
        Returns:
            (list of image tensors, list of texts), reached_end
        """
        batch, reachedEnd = loader.nextBatch()
        if not batch or reachedEnd:
            return ([], []), True

        imgTensors = []
        textTensors = []

        for imgPath, text in batch:
            # Load & decode image
            imgRaw = tf.io.read_file(imgPath)
            img = tf.image.decode_image(imgRaw, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]

            imgTensors.append(img)

            # Convert text to tf.Tensor
            textTensors.append(string_to_tensor(text))
            # textTensors.append(tf.convert_to_tensor(text, dtype=tf.string))
            # textTensors[-1] = tf.expand_dims(textTensors[-1], axis=0)

        # Stack into batch tensors
        imgBatch = tf.stack(imgTensors)       # (batch_size, H, W, 3)
        textBatch = tf.stack(textTensors)     # (batch_size, 256)

        return (imgBatch, textBatch), reachedEnd

    def saveModels(self):
        """
        Save the embedder and extractor models to the output directory.
        """
        self.embedder.save(f"{self.outputDir}/embedder.h5")
        self.extractor.save(f"{self.outputDir}/extractor.h5")
        print(f"Models saved to {self.outputDir}")

    def saveTrainLosses(self, epoch, step, losses):
        """
        Save the training loss history to a file.
        Is a csv file with columns: timestamp,epoch,step,total_loss,imperceptibility_loss,robustness_loss,extraction_loss
        """
        with open(f"{self.outputDir}/{Trainer.TrainLossesFilename()}", 'a') as f:
            line = f"{time.time()},{epoch},{step},{','.join([f'{loss:.4f}' for loss in losses])}\n"
            f.write(line)

    def saveValLosses(self, epoch, step, losses):
        """
        Save the validation loss history to a file.
        Is a csv file with columns: timestamp,epoch,step,total_loss,imperceptibility_loss,robustness_loss,extraction_loss
        """
        with open(f"{self.outputDir}/{Trainer.ValidationLossesFilename()}", 'a') as f:
            # join the losses with commas
            line = f"{time.time()},{epoch},{step},{','.join([f'{loss:.4f}' for loss in losses])}\n"
            f.write(line)

    @tf.function(reduce_retracing=True)
    def trainStep(self, imgs, wms):
        with tf.GradientTape() as tape:
            # Embed
            watermarkedImgs = self.embedder(imgs, wms)
            # Extract
            wmPreds = self.extractor(watermarkedImgs)
            # Compute loss
            totalLoss, L_imp, L_rob, L_ext = self.lossFn(imgs, watermarkedImgs, wms, wmPreds)

        grads = tape.gradient(totalLoss, self.embedder.trainable_variables + self.extractor.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.embedder.trainable_variables + self.extractor.trainable_variables))

        return totalLoss, L_imp, L_rob, L_ext

    @tf.function(reduce_retracing=True)
    def valStep(self, imgs, wms):
        # Embed
        watermarkedImgs = self.embedder(imgs, wms)
        # Extract
        wmPreds = self.extractor(watermarkedImgs)
        # Compute loss
        totalLoss, L_imp, L_rob, L_ext = self.lossFn(imgs, watermarkedImgs, wms, wmPreds)

        return totalLoss, L_imp, L_rob, L_ext

    def train(self, epochs=10):
        for epoch in tqdm(range(epochs), desc="Epoch"):
            # Training step
            trainLoop = tqdm(range(self.trainLoader.totalBatches), desc="Train", leave=False)
            losses = [[], [], [], []]
            for step in trainLoop:
                (imgs, wms), done = self._prepareBatch(self.trainLoader)
                loss, L_imp, L_rob, L_ext = self.trainStep(imgs, wms)
                trainLoop.set_postfix(total_loss=loss.numpy(), L_imp=L_imp.numpy(), L_rob=L_rob.numpy(), L_ext=L_ext.numpy())
                self.saveTrainLosses(epoch, step, [loss.numpy(), L_imp.numpy(), L_rob.numpy(), L_ext.numpy()])
                losses[0].append(loss.numpy())
                losses[1].append(L_imp.numpy())
                losses[2].append(L_rob.numpy())
                losses[3].append(L_ext.numpy())
                if done: break  # If we reached the end of the dataset, stop training

            tqdm.write(f"Saving models for epoch {epoch}...")
            tqdm.write(f"Training results for epoch {epoch}:")
            tqdm.write(f"Total loss: {np.mean(losses[0]):.4f}, L_imp: {np.mean(losses[1]):.4f}, L_rob: {np.mean(losses[2]):.4f}, L_ext: {np.mean(losses[3]):.4f}")
            logger.info(f"Epoch {epoch} completed. Total loss: {np.mean(losses[0]):.4f}, L_imp: {np.mean(losses[1]):.4f}, L_rob: {np.mean(losses[2]):.4f}, L_ext: {np.mean(losses[3]):.4f}")
            self.saveModels()

            # Validation step
            valLoop = tqdm(range(self.valLoader.totalBatches), desc="Validation", leave=False)
            losses = [[], [], [], []]
            for step in valLoop:
                (imgs, wms), done = self._prepareBatch(self.valLoader)
                loss, L_imp, L_rob, L_ext = self.valStep(imgs, wms)
                valLoop.set_postfix(total_loss=loss.numpy(), L_imp=L_imp.numpy(), L_rob=L_rob.numpy(), L_ext=L_ext.numpy())
                self.saveValLosses(epoch, step, [loss.numpy(), L_imp.numpy(), L_rob.numpy(), L_ext.numpy()])
                losses[0].append(loss.numpy())
                losses[1].append(L_imp.numpy())
                losses[2].append(L_rob.numpy())
                losses[3].append(L_ext.numpy())
                if done: break # If we reached the end of the dataset, stop validation

            tqdm.write(f"Validation results for epoch {epoch}:")
            tqdm.write(f"Total loss: {np.mean(losses[0]):.4f}, L_imp: {np.mean(losses[1]):.4f}, L_rob: {np.mean(losses[2]):.4f}, L_ext: {np.mean(losses[3]):.4f}")
            logger.info(f"Epoch {epoch} validation completed. Total loss: {np.mean(losses[0]):.4f}, L_imp: {np.mean(losses[1]):.4f}, L_rob: {np.mean(losses[2]):.4f}, L_ext: {np.mean(losses[3]):.4f}")
