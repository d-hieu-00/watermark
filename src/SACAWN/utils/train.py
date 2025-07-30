
# External
import sys
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
from utils.config import Configuration
from utils.load import loadImageTensor, loadStringTensor

logger = logging.getLogger(__name__)

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

    def __init__(self, config: Configuration):
        self.config         = config
        self.embedder       = None
        self.extractor      = None
        self.optimizer      = None
        self.trainLoader    = None
        self.valLoader      = None
        self.lossFn         = None
        self.outputDir      = self.config.outputPath
        self.wmMaxLen       = self.config.watermarkMaxLength
        self.wmVocabSize    = self.config.watermarkVocabSize
        self.imageTrainSize = config.imageTrainSize
        self.optimizerName  = self.config.optimizer
        self.learningRate   = self.config.learningRate
        self._setup()

        # Check if loss history files exist, if not create it
        self._checkLossHistoryFile(Trainer.TrainLossesFilename())
        self._checkLossHistoryFile(Trainer.ValidationLossesFilename())

    def _setup(self):
        # Prepare loaders
        self.trainLoader = DataLoader(
            imageDir=self.config.trainDatasetImageDir,
            watermarkFile=self.config.trainDatasetWatermarkFile,
            batchSize=self.config.trainDatasetBatchSize
        )
        self.valLoader = DataLoader(
            imageDir=self.config.validationDatasetImageDir,
            watermarkFile=self.config.validationDatasetWatermarkFile,
            batchSize=self.config.validationDatasetBatchSize
        )
        logger.info(f"Train Loader: {self.trainLoader.len()} samples, Batch Size: {self.trainLoader.batchSize}")
        logger.info(f"Validation Loader: {self.valLoader.len()} samples, Batch Size: {self.valLoader.batchSize}")
        logger.info(f"Watermark Max Length: {self.wmMaxLen}, Vocab Size: {self.wmVocabSize}")
        logger.info(f"Output Directory: {self.outputDir}")

        # Prepare models and optimizer, loss function
        from modules.model import WatermarkEmbedderModel, WatermarkExtractorModel
        from modules.lossfn import SACAWNLoss
        from tensorflow.python.keras.optimizer_v2.adam import Adam

        # Initialize models and loss function
        self.embedder    = WatermarkEmbedderModel(self.wmMaxLen, self.wmVocabSize).build()
        self.extractor   = WatermarkExtractorModel(self.wmMaxLen, self.wmVocabSize).build()
        self.lossFn      = SACAWNLoss(self.config.lossImperceptibilityWeight, self.config.lossRobustnessWeight, self.config.lossExtractionWeight)
        self.optimizer   = Adam(learning_rate=self.learningRate)
        if isinstance(self.optimizerName, str) and self.optimizerName.lower() == "adam":
            pass
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizerName}. Supported: 'adam'.")

        logger.info(f"Models initialized: {self.embedder.name} and {self.extractor.name}")
        logger.info(f"Optimizer: {self.optimizerName}, Learning Rate: {self.learningRate}")
        logger.info(f"Loss Function: {self.lossFn.name}")

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
            imgTensors.append(loadImageTensor(imgPath, self.imageTrainSize))  # Convert image to tf.Tensor
            textTensors.append(loadStringTensor(text, self.wmMaxLen))  # Convert text to tf.Tensor

        # Stack into batch tensors
        imgBatch = tf.stack(imgTensors)       # (batch_size, H, W, 3)
        textBatch = tf.stack(textTensors)     # (batch_size, 256)

        return (imgBatch, textBatch), reachedEnd

    def saveModels(self, printFn):
        """
        Save the embedder and extractor models to the output directory.
        """
        self.embedder.save(f"{self.outputDir}/embedder.h5", save_format='h5')
        self.extractor.save(f"{self.outputDir}/extractor.h5", save_format='h5')
        if printFn:
            printFn(f"Embedder model saved to {self.outputDir}/embedder.h5")
            printFn(f"Extractor model saved to {self.outputDir}/extractor.h5")

    def saveTrainLosses(self, epoch, step, losses):
        """
        Save the training loss history to a file.
        Is a csv file with columns: timestamp,epoch,step,total_loss,imperceptibility_loss,robustness_loss,extraction_loss
        """
        with open(f"{self.outputDir}/{Trainer.TrainLossesFilename()}", 'a') as f:
            line = f"{int(time.time())},{epoch},{step},{','.join([f'{loss:.4f}' for loss in losses])}\n"
            f.write(line)

    def saveValLosses(self, epoch, step, losses):
        """
        Save the validation loss history to a file.
        Is a csv file with columns: timestamp,epoch,step,total_loss,imperceptibility_loss,robustness_loss,extraction_loss
        """
        with open(f"{self.outputDir}/{Trainer.ValidationLossesFilename()}", 'a') as f:
            # join the losses with commas
            line = f"{int(time.time())},{epoch},{step},{','.join([f'{loss:.4f}' for loss in losses])}\n"
            f.write(line)

    @tf.function(reduce_retracing=True)
    def trainStep(self, imgs, wms):
        with tf.GradientTape() as tape:
            # Embed
            embImgs = self.embedder((imgs, wms))
            # Extract
            wmPreds = self.extractor(embImgs)
            # Compute loss
            totalLoss, L_imp, L_rob, L_ext = self.lossFn(imgs, embImgs, wms, wmPreds)

        grads = tape.gradient(totalLoss, self.embedder.trainable_variables + self.extractor.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.embedder.trainable_variables + self.extractor.trainable_variables))

        return totalLoss, L_imp, L_rob, L_ext

    @tf.function(reduce_retracing=True)
    def valStep(self, imgs, wms):
        # Embed
        watermarkedImgs = self.embedder((imgs, wms))
        # Extract
        wmPreds = self.extractor(watermarkedImgs)
        # Compute loss
        totalLoss, L_imp, L_rob, L_ext = self.lossFn(imgs, watermarkedImgs, wms, wmPreds)

        return totalLoss, L_imp, L_rob, L_ext

    def train(self, epochs=10):
        for epoch in tqdm(range(epochs), file=sys.stdout, desc="Epoch"):
            # Training step
            trainLoop = tqdm(range(self.trainLoader.totalBatches), file=sys.stdout, desc="+ Train", leave=False)
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

                if step % 2000 == 0 and step != 0:
                    tqdm.write(f"Saving models for each 2000 steps (step={step})...")
                    self.saveModels(printFn=tqdm.write)

            tqdm.write(f"Saving models for epoch {epoch}...")
            tqdm.write(f"Training results for epoch {epoch}:")
            tqdm.write(f"Total loss: {np.mean(losses[0]):.4f}, L_imp: {np.mean(losses[1]):.4f}, L_rob: {np.mean(losses[2]):.4f}, L_ext: {np.mean(losses[3]):.4f}")
            logger.info(f"Epoch {epoch} completed. Total loss: {np.mean(losses[0]):.4f}, L_imp: {np.mean(losses[1]):.4f}, L_rob: {np.mean(losses[2]):.4f}, L_ext: {np.mean(losses[3]):.4f}")
            self.saveModels(printFn=tqdm.write)

            # Validation step
            valLoop = tqdm(range(self.valLoader.totalBatches), file=sys.stdout, desc="+ Validation", leave=False)
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
