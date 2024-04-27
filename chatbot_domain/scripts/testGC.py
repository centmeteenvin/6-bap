import gc

import torch
import chatbot_domain.benchmark
from chatbot_domain import logger
from chatbot_domain.chatbot import ChatBotBuilder, DIPDomainGuard
from chatbot_domain.transformers import ModelBuilder

logger.setLevel("INFO")
# None Finetuned Model without RAG
logger.info(f"The starting memory stats are: {torch.cuda.mem_get_info()}")
logger.info("Loading big GPU model")
chatbot = ChatBotBuilder.model(
    ModelBuilder().modelName('mistralai/Mixtral-8x7B-Instruct-v0.1').deviceMap("auto").shouldQuantize(True).build(adapted=False)
    ).benchmarkGuard().domainGuard(DIPDomainGuard).build()
logger.info(f"The used memory is {torch.cuda.mem_get_info}")
chatbot.delete()
logger.info(f"The memory after calling chatbot.delete(): {torch.cuda.mem_get_info}")
del chatbot
logger.info(f"The memory after calling del chatbot: {torch.cuda.mem_get_info()}")
gc.collect()
logger.info(f"The memory after calling gc: {torch.cuda.mem_get_info()}")
torch.cuda.empty_cache()
logger.info(f"The memory after calling empty_cache: {torch.cuda.mem_get_info()}")