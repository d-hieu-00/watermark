import argparse
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
    logger.info(f"Training Epochs: {config.trainingEpochs}")

    stdErrLogFile = open(config.stdErrLogFile, 'w')
    import os; os.dup2(stdErrLogFile.fileno(), 2)  # Redirect stderr to stdErrLogFile for logging

    from utils.train import Trainer
    trainer = Trainer(config)
    trainer.train(epochs=config.trainingEpochs)
