import argparse
import os
from chatbot_domain.data import parseData, createDataSet, loadDataSetFromDisk
from chatbot_domain import logger
from . import *

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--parse-data", action="store_true", help="If we should parse our data or not")
    parser.add_argument("--file", help="File we read our data from", default="./DIP-TB.pdf")
    parser.add_argument("--level", help="logging level", default="INFO")
    parser.add_argument("--store-dataset", help="the path to store the dataset if set", default=None)
    parser.add_argument("--load-dataset", help= "Dataset you load from", default=None)
    
    args = parser.parse_args()
    
    logger.setLevel(args.level)
    dpr : DPR = FacebookDPR()

    dataset = None
    if args.parse_data:
        parsedData = parseData(args.file)
        dataset = createDataSet(parsedData, 0)
        logger.info("Encoding dataset in preparation for FAISS index")
        dataset = dataset.map(lambda set : {'embeddings' : dpr.encodeContext(set['sentence'])})
        logger.info("Creating FAISS index for dataset")
        dataset['train'].add_faiss_index(column='embeddings')
        
        if args.store_dataset is not None:
            logger.info(f"Storing dataset to {args.store_dataset}")
            dataset['train'].save_faiss_index('embeddings', file=f"{args.store_dataset}/faiss.index")
            dataset["train"	].drop_index('embeddings')
            dataset.save_to_disk(args.store_dataset)
            dataset["train"].load_faiss_index('embeddings', file=f"{args.store_dataset}/faiss.index")
    elif args.load_dataset is None:
        logger.error("You need to give the --load-dataset argument")
        exit()
    else:
        dataset = loadDataSetFromDisk(args.load_dataset)
        logger.info("loading indices into dataset")
        dataset["train"].load_faiss_index('embeddings',file= f"{args.load_dataset}/faiss.index")
    os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
    userInput = input('> ')
    while userInput != 'exit':
        questionEmbedding = dpr.encodeQuestion(userInput)
        scores, results = dataset['train'].get_nearest_examples(index_name= 'embeddings',query= questionEmbedding, k=5)
        print(results.keys())
        for score, result, paragraph in zip(reversed(scores), reversed(results['sentence']), reversed(results['paragraph'])):
            print(f"retrieved sentence: {(result)}\nfrom paragraph: {paragraph}\nscore: {score}")  
        userInput = input('> ')
if __name__ == '__main__' :
    main()