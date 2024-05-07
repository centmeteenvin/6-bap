import gc
from statistics import mean
from chatbot_domain.benchmark import JsonQuestionParser, Benchmarker, AlwaysATestSubject, NLPTestSubject, HumanTestSubject, TestScore
from chatbot_domain.chatbot import ChatBotModifier, OpenAIChatBot, DIPDomainGuard, ChatBotBuilder, RAGBuilder
from chatbot_domain.rag import VectorRetriever, FacebookDPR
from chatbot_domain.transformer import ModelBuilder
from chatbot_domain.data import loadDataSetFromDisk
from chatbot_domain import logger
import os
import datetime
import torch
import json
from random import seed

logger.setLevel("INFO")
parser = JsonQuestionParser('questions.json')
questions = parser.parse()

def runTest(scoreDirectory: str, testRunName: str, benchmarker: Benchmarker, summary : dict,samples: int = 3, delay = 0) -> list[TestScore]:
    # return []
    
    scores : list[TestScore] =  []
    logger.info(f"Starting tests for {testRunName}")
    summarySection = {
        'scores': []
    }
    
    for i in range(samples):
        seed(i*25 + 64)
        logger.info(f"Run {i+1}/{samples}")
        result = benchmarker.evaluate(delaySeconds=delay)
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







# BIG TESTRUN 1. Only local models.
# test base model
# test model with RAG 2k paragraph level
# test model with RAG 4k paragraph level
# test model with RAG 2k page level
# test model with RAG 4k page level

# # Mistral7B 
# No RAG
# chatbot = ChatBotBuilder.model(
#     ModelBuilder().modelName('mistralai/Mistral-7B-Instruct-v0.2').deviceMap("auto").shouldQuantize(True).build(adapted=False)
#     ).benchmarkGuard().domainGuard(DIPDomainGuard).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# testName = "mistral7B"
# scores.extend(runTest(testRunDirectory, "mistral7B", benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

# # with RAG 2k paragraph
# chatbot = ChatBotBuilder.model(
#     ModelBuilder().modelName('mistralai/Mistral-7B-Instruct-v0.2').deviceMap("auto").shouldQuantize(True).build(adapted=False)
#     ).benchmarkGuard().domainGuard(DIPDomainGuard).rag(
#         RAGBuilder.fromDatasetDisk('ragData.set').vectorRetriever(FacebookDPR()), contextLength= 2048
#     ).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, "mistral-7B-RAG2kparagraph", benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

# # with RAG 4k paragraph
# chatbot = ChatBotBuilder.model(
#     ModelBuilder().modelName('mistralai/Mistral-7B-Instruct-v0.2').deviceMap("auto").shouldQuantize(True).build(adapted=False)
#     ).benchmarkGuard().domainGuard(DIPDomainGuard).rag(
#         RAGBuilder.fromDatasetDisk('ragData.set').vectorRetriever(FacebookDPR()), contextLength= 4096
#     ).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, "mistral-7B-RAG4kparagraph", benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

# # with RAG 2k page
# chatbot = ChatBotBuilder.model(
#     ModelBuilder().modelName('mistralai/Mistral-7B-Instruct-v0.2').deviceMap("auto").shouldQuantize(True).build(adapted=False)
#     ).benchmarkGuard().domainGuard(DIPDomainGuard).rag(
#         RAGBuilder.fromDatasetDisk('ragDataPage.set').vectorRetriever(FacebookDPR()), contextLength= 2048
#     ).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, "mistral-7B-RAG2kpage", benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

# # with RAG 4k page
# chatbot = ChatBotBuilder.model(
#     ModelBuilder().modelName('mistralai/Mistral-7B-Instruct-v0.2').deviceMap("auto").shouldQuantize(True).build(adapted=False)
#     ).benchmarkGuard().domainGuard(DIPDomainGuard).rag(
#         RAGBuilder.fromDatasetDisk('ragDataPage.set').vectorRetriever(FacebookDPR()), contextLength= 4096
#     ).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, "mistral-7B-RAG4kpage", benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

# ## Mistral7B finetuned
# # without RAG
# chatbot = ChatBotBuilder.model(
#     ModelBuilder().modelName('vincentverbergt/Mistral7B-DIP').deviceMap("auto").shouldQuantize(True).build(adapted=True)
#     ).benchmarkGuard().domainGuard(DIPDomainGuard).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, "mistral7B-ft", benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

# # RAG2k Sentence
# chatbot = ChatBotBuilder.model(
#     ModelBuilder().modelName('vincentverbergt/Mistral7B-DIP').deviceMap("auto").shouldQuantize(True).build(adapted=True)
#     ).benchmarkGuard().domainGuard(DIPDomainGuard).rag(
#         RAGBuilder.fromDatasetDisk('ragData.set').vectorRetriever(FacebookDPR()), contextLength= 2048
#     ).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, "mistral7B-ft-RAG2kparagraph", benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

# ## Mixtral 8x7b

# # No RAG
# chatbot = ChatBotBuilder.model(
#     ModelBuilder().modelName('mistralai/Mixtral-8x7B-Instruct-v0.1').deviceMap("auto").shouldQuantize(True).build(adapted=False)
#     ).benchmarkGuard().domainGuard(DIPDomainGuard).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, "mixtral8x7B", benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

# # with RAG 2k paragraph
# chatbot = ChatBotBuilder.model(
#     ModelBuilder().modelName('mistralai/Mixtral-8x7B-Instruct-v0.1').deviceMap("auto").shouldQuantize(True).build(adapted=False)
#     ).benchmarkGuard().domainGuard(DIPDomainGuard).rag(
#         RAGBuilder.fromDatasetDisk('ragData.set').vectorRetriever(FacebookDPR()), contextLength= 2048
#     ).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, "mixtral-8x7B-RAG2kparagraph", benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

# # with RAG 4k paragraph
# chatbot = ChatBotBuilder.model(
#     ModelBuilder().modelName('mistralai/Mixtral-8x7B-Instruct-v0.1').deviceMap("auto").shouldQuantize(True).build(adapted=False)
#     ).benchmarkGuard().domainGuard(DIPDomainGuard).rag(
#         RAGBuilder.fromDatasetDisk('ragData.set').vectorRetriever(FacebookDPR()), contextLength= 4096
#     ).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, "mixtral-8x7B-RAG4kparagraph", benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

# # with RAG 2k page
# chatbot = ChatBotBuilder.model(
#     ModelBuilder().modelName('mistralai/Mixtral-8x7B-Instruct-v0.1').deviceMap("auto").shouldQuantize(True).build(adapted=False)
#     ).benchmarkGuard().domainGuard(DIPDomainGuard).rag(
#         RAGBuilder.fromDatasetDisk('ragDataPage.set').vectorRetriever(FacebookDPR()), contextLength= 2048
#     ).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, "mixtral-8x7B-RAG2kpage", benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

# # with RAG 4k page
# chatbot = ChatBotBuilder.model(
#     ModelBuilder().modelName('mistralai/Mixtral-8x7B-Instruct-v0.1').deviceMap("auto").shouldQuantize(True).build(adapted=False)
#     ).benchmarkGuard().domainGuard(DIPDomainGuard).rag(
#         RAGBuilder.fromDatasetDisk('ragDataPage.set').vectorRetriever(FacebookDPR()), contextLength= 4096
#     ).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, "mixtral-8x7B-RAG4kpage", benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()


# ## llama3 8x7b

# # No RAG
# chatbot = ChatBotBuilder.model(
#     ModelBuilder().modelName('meta-llama/Meta-Llama-3-70B-Instruct').deviceMap("auto").shouldQuantize(True).build(adapted=False)
#     ).benchmarkGuard().domainGuard(DIPDomainGuard).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, "llama-3-70B", benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

# # with RAG 2k paragraph
# chatbot = ChatBotBuilder.model(
#     ModelBuilder().modelName('meta-llama/Meta-Llama-3-70B-Instruct').deviceMap("auto").shouldQuantize(True).build(adapted=False)
#     ).benchmarkGuard().domainGuard(DIPDomainGuard).rag(
#         RAGBuilder.fromDatasetDisk('ragData.set').vectorRetriever(FacebookDPR()), contextLength= 2048
#     ).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, "llama-3-70B-RAG2kparagraph", benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

# # with RAG 4k paragraph
# chatbot = ChatBotBuilder.model(
#     ModelBuilder().modelName('meta-llama/Meta-Llama-3-70B-Instruct').deviceMap("auto").shouldQuantize(True).build(adapted=False)
#     ).benchmarkGuard().domainGuard(DIPDomainGuard).rag(
#         RAGBuilder.fromDatasetDisk('ragData.set').vectorRetriever(FacebookDPR()), contextLength= 4096
#     ).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, "llama-3-70B-RAG4kparagraph", benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

# # with RAG 2k page
# chatbot = ChatBotBuilder.model(
#     ModelBuilder().modelName('meta-llama/Meta-Llama-3-70B-Instruct').deviceMap("auto").shouldQuantize(True).build(adapted=False)
#     ).benchmarkGuard().domainGuard(DIPDomainGuard).rag(
#         RAGBuilder.fromDatasetDisk('ragDataPage.set').vectorRetriever(FacebookDPR()), contextLength= 2048
#     ).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, "llama-3-70B-RAG2kpage", benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

# # with RAG 4k page
# chatbot = ChatBotBuilder.model(
#     ModelBuilder().modelName('meta-llama/Meta-Llama-3-70B-Instruct').deviceMap("auto").shouldQuantize(True).build(adapted=False)
#     ).benchmarkGuard().domainGuard(DIPDomainGuard).rag(
#         RAGBuilder.fromDatasetDisk('ragDataPage.set').vectorRetriever(FacebookDPR()), contextLength= 4096
#     ).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, "llama-3-70B-RAG4kpage", benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()


## TEST 2
# GPT-3.5 Turbo test.
# 2k 4k 8k windows in page en paragraph mode
# Baseline
# testName = 'GPT-3.5-Turbo'
# chatbot = ChatBotBuilder.openAI().benchmarkGuard().domainGuard(DIPDomainGuard).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, testName, benchmarker, summary))

# # WINDOWSIZE 2k
# contextLength = 2048
# # Paragraph
# isParagraph = True
# testName = 'GPT-3.5-Turbo-RAG2kparagraph' 
# chatbot = ChatBotBuilder.openAI().benchmarkGuard().domainGuard(DIPDomainGuard).rag(
#     RAGBuilder.fromDatasetDisk('./ragData.set' if isParagraph else './ragDataPage.set').vectorRetriever(FacebookDPR()), contextLength= contextLength
# ).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, testName, benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

# # Page
# isParagraph = False
# testName = 'GPT-3.5-Turbo-RAG2kpage' 
# chatbot = ChatBotBuilder.openAI().benchmarkGuard().domainGuard(DIPDomainGuard).rag(
#     RAGBuilder.fromDatasetDisk('./ragData.set' if isParagraph else './ragDataPage.set').vectorRetriever(FacebookDPR()), contextLength= contextLength
# ).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, testName, benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

# # WINDOWSIZE 4k
# contextLength = 4096
# # Paragraph
# isParagraph = True
# testName = 'GPT-3.5-Turbo-RAG4kparagraph' 
# chatbot = ChatBotBuilder.openAI().benchmarkGuard().domainGuard(DIPDomainGuard).rag(
#     RAGBuilder.fromDatasetDisk('./ragData.set' if isParagraph else './ragDataPage.set').vectorRetriever(FacebookDPR()), contextLength= contextLength
# ).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, testName, benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

# # Page
# isParagraph = False
# testName = 'GPT-3.5-Turbo-RAG4kpage' 
# chatbot = ChatBotBuilder.openAI().benchmarkGuard().domainGuard(DIPDomainGuard).rag(
#     RAGBuilder.fromDatasetDisk('./ragData.set' if isParagraph else './ragDataPage.set').vectorRetriever(FacebookDPR()), contextLength= contextLength
# ).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, testName, benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

# # WINDOWSIZE 8k
# contextLength = 8192
# # Paragraph
# isParagraph = True
# testName = 'GPT-3.5-Turbo-RAG8kparagraph' 
# chatbot = ChatBotBuilder.openAI().benchmarkGuard().domainGuard(DIPDomainGuard).rag(
#     RAGBuilder.fromDatasetDisk('./ragData.set' if isParagraph else './ragDataPage.set').vectorRetriever(FacebookDPR()), contextLength= contextLength
# ).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, testName, benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

# # Page
# isParagraph = False
# testName = 'GPT-3.5-Turbo-RAG8kpage' 
# chatbot = ChatBotBuilder.openAI().benchmarkGuard().domainGuard(DIPDomainGuard).rag(
#     RAGBuilder.fromDatasetDisk('./ragData.set' if isParagraph else './ragDataPage.set').vectorRetriever(FacebookDPR()), contextLength= contextLength
# ).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, testName, benchmarker, summary))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

## Test 3
# payed  closed source models, 1 run @ 2kPageRAG

# GPT-3.5-Turbo
#
# chatbot = ChatBotBuilder.openAI().benchmarkGuard().domainGuard(DIPDomainGuard).rag(
#     RAGBuilder.fromDatasetDisk('./ragDataPage.set').vectorRetriever(FacebookDPR()), 2048
# ).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, 'GPT-3.5-Turbo', benchmarker, summary, samples=1, delay=30))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

# # GPT-4-Turbo
# #
# chatbot = ChatBotBuilder.openAI('gpt-4-turbo-2024-04-09').benchmarkGuard().domainGuard(DIPDomainGuard).rag(
#     RAGBuilder.fromDatasetDisk('./ragDataPage.set').vectorRetriever(FacebookDPR()), 2048
# ).build()
# benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
# scores.extend(runTest(testRunDirectory, 'GPT-4-Turbo', benchmarker, summary,samples=1, delay=30))
# del benchmarker._testSubject
# del benchmarker
# del chatbot
# gc.collect()
# torch.cuda.empty_cache()

# Claude-3-opus
#
chatbot = ChatBotBuilder.anthropic('claude-3-opus-20240229').benchmarkGuard().domainGuard(DIPDomainGuard).rag(
    RAGBuilder.fromDatasetDisk('./ragDataPage.set').vectorRetriever(FacebookDPR()), 2048
).build()
benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
scores.extend(runTest(testRunDirectory, 'claude-3-opus', benchmarker, summary, samples=1, delay=30))
del benchmarker._testSubject
del benchmarker
del chatbot
gc.collect()
torch.cuda.empty_cache()

# Claude-3-sonnet
#
chatbot = ChatBotBuilder.anthropic('claude-3-sonnet-20240229').benchmarkGuard().domainGuard(DIPDomainGuard).rag(
    RAGBuilder.fromDatasetDisk('./ragDataPage.set').vectorRetriever(FacebookDPR()), 2048
).build()
benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
scores.extend(runTest(testRunDirectory, 'claude-3-sonnet', benchmarker, summary, samples=1, delay=30))
del benchmarker._testSubject
del benchmarker
del chatbot
gc.collect()
torch.cuda.empty_cache()

# Claude-3-sonnet
#
chatbot = ChatBotBuilder.anthropic('claude-3-haiku-20240307').benchmarkGuard().domainGuard(DIPDomainGuard).rag(
    RAGBuilder.fromDatasetDisk('./ragDataPage.set').vectorRetriever(FacebookDPR()), 2048
).build()
benchmarker = Benchmarker(NLPTestSubject(chatbot), questions)
scores.extend(runTest(testRunDirectory, 'claude-3-haiku', benchmarker, summary, samples=1, delay=30))
del benchmarker._testSubject
del benchmarker
del chatbot
gc.collect()
torch.cuda.empty_cache()


print(scores)
print(summary)
with open(f"{testRunDirectory}/summary.json", 'w') as file:
    jsonText = json.dumps(summary)
    file.write(jsonText)