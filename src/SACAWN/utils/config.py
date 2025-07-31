import json
import logging
import time


class Configuration:
    _logger = logging.getLogger(__name__)
    def __init__(self, configFile="config.json", setupLogging=True):
        self._timestamp = time.strftime("%Y%m%d-%H%M%S")
        self._setupLogging = setupLogging
        # Load json configuration file to memory
        try:
            with open(configFile, "r") as f:
                self._configJson = json.load(f)
            self._configFile = configFile
            self._logger.info(f"Configuration loaded from {configFile}")
            print(f"Configuration loaded from {configFile} - timestamp: {self._timestamp}")
        except FileNotFoundError:
            self._logger.error(f"Configuration file {configFile} not found.")
            raise

        # Make output path if it does not exist
        if self.outputPath:
            import os
            if not os.path.exists(self.outputPath):
                os.makedirs(self.outputPath)
                print(f"Created output directory: {self.outputPath}")

        # Set logging level based on configuration
        if self._setupLogging:
            logging.basicConfig(filename=self.loggingFile, level=self.loggingLevel.upper(),
                                format='%(asctime)s - %(levelname)s - %(message)s')
            print(f"Logging initialized with level: {self.loggingLevel} and file: {self.loggingFile}")

        # Set the runtime environment
        envs = self.get("environment", default={})
        if envs:
            for key, value in envs.items():
                import os
                os.environ[key] = str(value)
                print(f"Set environment variable {key} to {value}")

    def get(self, *keys, default=None):
        """
        Retrieve a configuration value using a sequence of keys.
        Example: config.get("model", "embedded", "modelPath")
        """
        value = self._configJson
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            self._logger.error(f"Configuration key path {' -> '.join(map(str, keys))} not found.")
            raise
    
    @property
    def modelEmbeddedPath(self):
        """
        Get the model path for the embedded model.
        """
        return self.get("model", "embedded", "modelPath")

    @property
    def modelExtractorPath(self):
        """
        Get the model path for the extractor model.
        """
        return self.get("model", "extractor", "modelPath")

    @property
    def trainDatasetImageDir(self):
        """
        Get the dataset imageDir for the training dataset.
        """
        return self.get("dataset", "train", "imageDir")
    
    @property
    def trainDatasetBatchSize(self):
        """
        Get the batch size of the training dataset.
        """
        return self.get("dataset", "train", "batchSize")

    @property
    def trainDatasetWatermarkFile(self):
        """
        Get the watermark file path for the training dataset.
        """
        return self.get("dataset", "train", "watermarkFile")

    @property
    def validationDatasetImageDir(self):
        """
        Get the dataset imageDir for the validation dataset.
        """
        return self.get("dataset", "validation", "imageDir")
    
    @property
    def validationDatasetBatchSize(self):
        """
        Get the batch size of the validation dataset.
        """
        return self.get("dataset", "validation", "batchSize")

    @property
    def validationDatasetWatermarkFile(self):
        """
        Get the watermark file path for the validation dataset.
        """
        return self.get("dataset", "validation", "watermarkFile")

    @property
    def watermarkMaxLength(self):
        """
        Get the maximum length of the watermark.
        """
        return self.get("training", "watermarkMaxLength")
    
    @property
    def watermarkVocabSize(self):
        """
        Get the vocabulary size for the watermark.
        """
        return self.get("training", "watermarkVocabSize")

    @property
    def lossImperceptibilityWeight(self):
        """
        Get the weight for the imperceptibility loss.
        """
        return self.get("training", "loss", "imperceptibilityWeight")

    @property
    def lossRobustnessWeight(self):
        """
        Get the weight for the robustness loss.
        """
        return self.get("training", "loss", "robustnessWeight")

    @property
    def lossExtractionWeight(self):
        """
        Get the weight for the extraction loss.
        """
        return self.get("training", "loss", "extractionWeight")

    @property
    def trainingEpochs(self):
        """
        Get the number of training epochs.
        """
        return self.get("training", "epochs")

    @property
    def optimizer(self):
        """
        Get the optimizer used for training.
        """
        return self.get("training", "optimizer")

    @property
    def learningRate(self):
        """
        Get the learning rate for the optimizer.
        """
        return self.get("training", "learningRate", default=1e-4)

    @property
    def outputPath(self):
        """
        Get the output path for training results.
        """
        return f"{self.get("training", "output")}/{self._timestamp}"

    @property
    def loggingLevel(self):
        """
        Get the logging level for training.
        """
        return self.get("training", "logging", "logLevel")

    @property
    def loggingFile(self):
        """
        Get the logging file name for training.
        """
        if not self._setupLogging:
            return None
        return f"{self.outputPath}/{self.get("training", "logging", "logFile")}"

    @property
    def stdErrLogFile(self):
        """
        Get the standard error logging file name for training.
        """
        if not self._setupLogging:
            return None
        return f"{self.outputPath}/{self.get("training", "logging", "stdErrLogFile")}"

    @property
    def imageTrainSize(self):
        """
        Get the image train size (H, W)
        """
        h = self.get("training", "imageHeight", default=256)
        w = self.get("training", "imageWidth", default=256)
        return (h, w)