import gc

import torch
import chatbot_domain.benchmark
from chatbot_domain import logger
from chatbot_domain.chatbot import ChatBotBuilder, DIPDomainGuard
from chatbot_domain.transformers import ModelBuilder

def getCurrentMemory() -> str:
    free_memory, total_memory = torch.cuda.mem_get_info()
    allocated_memory = torch.cuda.memory_allocated()
    return f"""
Total:      {total_memory/1_000_000:20,.2f} MB
Free:       {free_memory/1_000_000:20,.2f} MB
allocated:  {allocated_memory/1_000_000:20,.2f} MB
"""

logger.setLevel("INFO")
# None Finetuned Model without RAG
logger.info(f"The starting memory stats are: {getCurrentMemory()}")
logger.info("Loading big GPU model")
chatbot = ChatBotBuilder.model(
    ModelBuilder().modelName('mistralai/Mixtral-8x7B-Instruct-v0.1').deviceMap("auto").shouldQuantize(True).build(adapted=False)
    ).benchmarkGuard().domainGuard(DIPDomainGuard).build()
logger.info(f"The used memory is {torch.cuda.memory_allocated}")
chatbot.delete()
logger.info(f"The memory after calling chatbot.delete(): {getCurrentMemory()}")
del chatbot
logger.info(f"The memory after calling del chatbot: {getCurrentMemory()}")
gc.collect()
logger.info(f"The memory after calling gc: {getCurrentMemory()}")
torch.cuda.empty_cache()
logger.info(f"The memory after calling empty_cache: {getCurrentMemory()}")