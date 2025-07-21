import argparse
import string
import logging
from utils.config import Configuration
from modules.loader import *

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configFile', required=False, type=str, default='./src/SACAWN/config.json', help='Path to the config file')
    args = parser.parse_args()

    # Load and print watermarks
    config = Configuration(args.configFile)
    logger.info(f"Configuration loaded: {config._configFile}")

    # Prepare loaders
    trainLoader = DataLoader(
        imageDir=config.trainDatasetImageDir,
        watermarkFile=config.trainDatasetWatermarkFile,
        batchSize=config.trainDatasetBatchSize
    )
    valLoader = DataLoader(
        imageDir=config.validationDatasetImageDir,
        watermarkFile=config.validationDatasetWatermarkFile,
        batchSize=config.validationDatasetBatchSize
    )
    logger.info(f"Train Loader: {trainLoader.len()} samples, Batch Size: {trainLoader.batchSize}")
    logger.info(f"Validation Loader: {valLoader.len()} samples, Batch Size: {valLoader.batchSize}")

    # Prepare models and optimizer, loss function
    from modules.model import WatermarkEmbedderModel, WatermarkExtractorModel
    from modules.lossfn import SACAWNLoss
    from tensorflow.python.keras.optimizer_v2.adam import Adam

    wmMaxLen      = config.watermarkMaxLength
    wmVocabSize   = config.watermarkVocabSize
    optimizerName = config.optimizer
    outputDir     = config.outputPath
    learningRate  = config.learningRate
    trainEpochs   = config.trainingEpochs

    logger.info(f"Watermark Max Length: {wmMaxLen}, Vocab Size: {wmVocabSize}")
    logger.info(f"Training Epochs: {trainEpochs}")
    logger.info(f"Output Directory: {outputDir}")

    # Initialize models and loss function
    embedder    = WatermarkEmbedderModel(wmMaxLen, wmVocabSize)
    extractor   = WatermarkExtractorModel(wmMaxLen, wmVocabSize)
    lossFn      = SACAWNLoss(config.lossImperceptibilityWeight, config.lossRobustnessWeight, config.lossExtractionWeight)
    optimizer   = Adam(learning_rate=learningRate)
    if isinstance(optimizerName, str) and optimizerName.lower() == "adam":
        pass
    else:
        raise ValueError(f"Unsupported optimizer: {optimizerName}. Supported: 'adam'.")
    
    logger.info(f"Models initialized: {embedder.name} and {extractor.name}")
    logger.info(f"Optimizer: {optimizerName}, Learning Rate: {learningRate}")
    logger.info(f"Loss Function: {lossFn.name}")

    from utils.train import Trainer
    trainer = Trainer(embedder, extractor, optimizer, trainLoader, valLoader, lossFn, outputDir)
    trainer.train(epochs=trainEpochs)
    # train(trainLoader, valLoader)