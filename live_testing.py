from rag_chatbot import (
    PDFTextSource,
    TextCleanup,
    AlphaNumericalTextTransformer,
    TrimTextTransformer,
    RAGAugment,
    DPRTextSearch,
    AnthropicResolver,
    AnthropicModels,
    OpenAIResolver,
    Chatbot,
)

source = PDFTextSource("./test/resources/attention_is_all_you_need.pdf")
cleanup = TextCleanup([TrimTextTransformer(), AlphaNumericalTextTransformer()])
search = DPRTextSearch(source, cleanup)
resolver = AnthropicResolver(AnthropicModels.CLAUDE_3_HAIKU.value)
# resolver = OpenAIResolver('gpt-3.5-turbo')
augment = RAGAugment(search, resolver, numberOfTokens=2048)
chatbot = Chatbot(resolver, augments=[augment])
chatbot.startConversation()