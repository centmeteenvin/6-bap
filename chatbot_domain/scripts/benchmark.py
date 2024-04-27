import gc
from statistics import mean
from chatbot_domain.benchmark import JsonQuestionParser, Benchmarker, AlwaysATestSubject, NLPTestSubject, HumanTestSubject, TestScore
from chatbot_domain.chatbot import ChatBotModifier, OpenAIChatBot, DIPDomainGuard, ChatBotBuilder, RAGBuilder
from chatbot_domain.rag import VectorRetriever, FacebookDPR
from chatbot_domain.transformers import ModelBuilder
from chatbot_domain.data import loadDataSetFromDisk
from chatbot_domain import logger
import os
import datetime
import torch
import json

logger.setLevel("INFO")
parser = JsonQuestionParser('questions.json')
questions = parser.parse()

def runTest(scoreDirectory: str, testRunName: str, benchmarker: Benchmarker, summary : dict,samples: int = 3) -> list[TestScore]:
    scores : list[TestScore] =  []
    logger.info(f"Starting tests for {testRunName}")
    summarySection = {
        'scores': []
    }
    
    for i in range(samples):
        logger.info(f"Run {i+1}/{samples}")
        result = benchmarker.evaluate()
        scores.append(result)
        result.save(f"{scoreDirectory}/{testRunName}")
        summarySection['scores'].append(result.total)
        print(f"{testRunName} run {i+1}:\n{result}")
    average = mean([score.average for score in scores])
    summarySection['average'] = average
    summary[testRunName] = summarySection
    print(f"Average score for {testRunName}: {average}")
    return scores
            
globalScoreDirectory = "./scores"            
testRunDirectory = None
i = 0
while testRunDirectory is None:
    proposal = f"{globalScoreDirectory}/{datetime.date.today().strftime('%d-%m-%Y')}-{i}"
    if not os.path.exists(proposal):
        testRunDirectory = proposal
        break
    i += 1
scores = []
summary = {}
# OpenAI models with RAG
# chatbot = ChatBotBuilder.openAI('gpt-4-turbo').benchmarkGuard().domainGuard(DIPDomainGuard).rag(
#     RAGBuilder.fromDatasetDisk('ragData.set').vectorRetriever(FacebookDPR()), contextLength = 1024 * 8
#     ).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)

# None Finetuned Model without RAG
chatbot = ChatBotBuilder.model(
    ModelBuilder().modelName('mistralai/Mixtral-8x7B-Instruct-v0.1').deviceMap("auto").shouldQuantize(True).build(adapted=False)
    ).benchmarkGuard().domainGuard(DIPDomainGuard).build()
benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
scores.extend(runTest(testRunDirectory, "mixtral-8*7B", benchmarker, summary))
del chatbot
gc.collect()
torch.cuda.empty_cache()

# None Finetuned Model with RAG
chatbot = ChatBotBuilder.model(
    ModelBuilder().modelName('mistralai/Mixtral-8x7B-Instruct-v0.1').deviceMap("auto").shouldQuantize(True).build(adapted=False)
    ).benchmarkGuard().domainGuard(DIPDomainGuard).rag(
        RAGBuilder.fromDatasetDisk('ragData.set').vectorRetriever(FacebookDPR()), contextLength= 2048
    ).build()
benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
scores.extend(runTest(testRunDirectory, "mixtral-8*7B-RAG2k", benchmarker, summary))
del chatbot
gc.collect()
torch.cuda.empty_cache()


# None Finetuned Model without RAG
chatbot = ChatBotBuilder.model(
    ModelBuilder().modelName('mistralai/Mistral-7B-Instruct-v0.2').deviceMap("auto").shouldQuantize(True).build(adapted=False)
    ).benchmarkGuard().domainGuard(DIPDomainGuard).build()
benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
testName = "mistral7B"
scores.extend(runTest(testRunDirectory, "mistral7B", benchmarker, summary))
del chatbot
gc.collect()
torch.cuda.empty_cache()

# None Finetuned Model with RAG
chatbot = ChatBotBuilder.model(
    ModelBuilder().modelName('mistralai/Mistral-7B-Instruct-v0.2').deviceMap("auto").shouldQuantize(True).build(adapted=False)
    ).benchmarkGuard().domainGuard(DIPDomainGuard).rag(
        RAGBuilder.fromDatasetDisk('ragData.set').vectorRetriever(FacebookDPR()), contextLength= 2048
    ).build()
benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
scores.extend(runTest(testRunDirectory, "mistral7B-RAG2k", benchmarker, summary))
del chatbot
gc.collect()
torch.cuda.empty_cache()

# Finetuned Model without RAG
chatbot = ChatBotBuilder.model(
    ModelBuilder().modelName('vincentverbergt/Mistral7B-DIP').deviceMap("auto").shouldQuantize(True).build(adapted=True)
    ).benchmarkGuard().domainGuard(DIPDomainGuard).build()
benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
scores.extend(runTest(testRunDirectory, "mistral7B-ft", benchmarker, summary))
del chatbot
gc.collect()
torch.cuda.empty_cache()

# Finetuned Model with RAG
chatbot = ChatBotBuilder.model(
    ModelBuilder().modelName('vincentverbergt/Mistral7B-DIP').deviceMap("auto").shouldQuantize(True).build(adapted=True)
    ).benchmarkGuard().domainGuard(DIPDomainGuard).rag(
        RAGBuilder.fromDatasetDisk('ragData.set').vectorRetriever(FacebookDPR()), contextLength= 2048
    ).build()
benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
scores.extend(runTest(testRunDirectory, "mistral7B-ft-RAG2k", benchmarker, summary))
del chatbot
gc.collect()
torch.cuda.empty_cache()


# Human Test subject
# name = input("name: ")
# benchmarker = Benchmarker(HumanTestSubject(name), questions)


print(scores)
print(summary)
with open(f"{testRunDirectory}/summary.json", 'w') as file:
    jsonText = json.dumps(scores)
    file.write(jsonText)