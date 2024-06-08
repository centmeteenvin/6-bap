# from rag_chatbot import (
#     PDFTextSource,
#     TextCleanup,
#     AlphaNumericalTextTransformer,
#     TrimTextTransformer,
#     RAGAugment,
#     DPRTextSearch,
#     AnthropicResolver,
#     AnthropicModels,
#     OpenAIResolver,
#     Chatbot,
# )

# source = PDFTextSource("./test/resources/attention_is_all_you_need.pdf")
# cleanup = TextCleanup([TrimTextTransformer(), AlphaNumericalTextTransformer()])
# search = DPRTextSearch(source, cleanup)
# resolver = AnthropicResolver(AnthropicModels.CLAUDE_3_HAIKU.value)
# # resolver = OpenAIResolver('gpt-3.5-turbo')
# augment = RAGAugment(search, resolver, numberOfTokens=2048)
# chatbot = Chatbot(resolver, augments=[augment])
# chatbot.startConversation()

from urllib import response
from openai import OpenAI, Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from rag_chatbot.secrets import Secrets
from transformers.pipelines import Conversation

client = OpenAI(api_key = Secrets.openAIKey())
result : Stream[ChatCompletionChunk] = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=Conversation("Hello my name is vincent, What is life like?"),
    stream=True
)

for chunk in result:
    print(chunk.choices[0].delta.content, end= '')
