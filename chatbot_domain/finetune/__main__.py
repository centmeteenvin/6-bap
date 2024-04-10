import argparse, logging
from chatbot_domain import logger, ch
from chatbot_domain.data import parseData, createDataSet, loadDataSetFromDisk
from chatbot_domain.transformers.tokenizer import Tokenizer
from chatbot_domain.settings import Settings
from chatbot_domain.transformers.model import Model
from chatbot_domain.chatbot.chatbot import ChatBot
from . import *

def main(
    dataFilePath: str,
    dataSetStorePath: str | None,
    dataSetLoadPath: str | None,
):  
    model = Model()
    if Settings.shouldTrain:
        dataSet = None
        if dateSetLoadPath is None:
            sentences, alineas = parseData(dataFilePath)
            logger.debug(alineas)
            logger.debug(sentences)
            dataSet = createDataSet(alineas, 0.05)
            dataSet = Tokenizer.get().encode(dataSet)
            if dataSetStorePath:
                logger.info(f"Storing dataset to {dataSetStorePath}")
                dataSet.save_to_disk(dataSetStorePath)
        else:
            logger.info("Loading dataset from disk")
            dataSet = loadDataSetFromDisk(dataSetLoadPath)
        logger.info(f"dataset: {dataSet}")
        trainer = Training(model, dataSet)
        trainer.train()
    if Settings.shouldChat:
        chatbot = ChatBot(model, Tokenizer.get())
        chatbot.run()
    

        
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help="The file to read from the data from", default="./DIP-TB.pdf")
    parser.add_argument('--level', type=str, help='Set the logging level')
    parser.add_argument('--store-dataset', type=str, help='Store parsed dataSet to string location', default=None)
    parser.add_argument('--load-dataset', type=str, help="Loads a tokenized dataset from disk", default=None)
    parser.add_argument('--name', type=str, help="The name of the model we want to fineTune", default="google/gemma-2b")
    parser.add_argument('--training-output', type=str, help="The training output directory", default="./training")
    parser.add_argument('--batch-size', type= int, help="The batchsize to train the model in", default=32)
    parser.add_argument('--epoch', type=int, help="Number of epochs to train on", default=3)
    parser.add_argument('--eval-batch-size', type=int, help="The batch-size multiplier for evaluation batch-size", default=2)
    parser.add_argument('--gradient-accumulator', type=int, help="The number of gradient accumulator steps, effectively increases batch size", default=2),
    parser.add_argument('--train', action='store_true', help="If the script should train/finetune the model or solely act as a chatbot")
    parser.add_argument('--model-path', type=str, help="If this is given it will load the model from the path instead of the name", default= None)
    parser.add_argument('--quantize', action='store_true', help="Quantizes the model, this is not allowed when --train is given")
    parser.add_argument('--auto-map', action='store_true', help="Should the model be auto mapped, otherwise it will be on cuda")
    parser.add_argument('--optimizer', type=str, help="The optimizer being used, defaults to adamw_torch_fused", default='adamw_torch_fused')
    parser.add_argument('--no-chat', action='store_true', help="If the program should chat with the bot")
    parser.add_argument('--model-save-path', type=str, help="Where to save the model once it is finished training", default="./models/unnamed")
    args = parser.parse_args()
    
    if args.level:
        logger.setLevel(args.level.upper())
    else:
        logger.setLevel(logging.DEBUG)
        
    dataFilePath= args.file
    dataSetStorePath = args.store_dataset
    dateSetLoadPath = args.load_dataset
    Settings.modelName = args.name
    from chatbot_domain.secrets import HUGGINGFACE_ACCESS
    Settings.accesToken = HUGGINGFACE_ACCESS
    Settings.trainingOutput = args.training_output
    Settings.batchSize = args.batch_size
    Settings.epochs = args.epoch
    Settings.evalBatchSizeMultiplier = args.eval_batch_size
    Settings.gradientAccumulatorMultiplier = args.gradient_accumulator
    Settings.shouldTrain = args.train
    Settings.modelPath = args.model_path
    Settings.shouldQuantize = args.quantize
    Settings.autoMap = args.auto_map
    Settings.optimizer = args.optimizer
    Settings.shouldChat = not args.no_chat
    Settings.modelSavePath = args.model_save_path
    
    if Settings.shouldTrain and Settings.shouldQuantize:
        logger.fatal("Quantization and train options were given, this is prohibited")
        exit(69)
    
    
    main(dataFilePath, dataSetStorePath, dateSetLoadPath)