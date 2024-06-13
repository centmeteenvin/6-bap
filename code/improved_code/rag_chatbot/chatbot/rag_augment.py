from math import floor
from transformers.pipelines import Conversation
from rag_chatbot.chatbot.resolver import Resolver
from rag_chatbot.similarity_search import TextSearch, QueryResult

from .augment import PromptAugment

class RAGAugment(PromptAugment):
    """
    This is an augmentation you can apply to any chatbot. The
    augmentConversation function can be called to augment to conversational
    object and append some context found in the textSource. A TextSearch Object
    is necessary for this functionality.
    """
    def __init__(self, textSearch : TextSearch, resolver : Resolver, numberOfTokens: int = 1024) -> None:
        super().__init__()
        assert isinstance(textSearch, TextSearch)
        assert numberOfTokens > 0
        assert isinstance(resolver, Resolver)
        self.search = textSearch
        self.numberOfTokens = numberOfTokens
        self.initialNGuess = 1 # We try to find the amount of results we need to ask from the search the reach the numberOfTokens
        self.resolver = resolver


    def augmentConversation(self, conversation: Conversation) ->tuple[Conversation, list[QueryResult]]:
        """
        Insert a system 'prompt' containing extra information equalling the
        amount of tokens specified in the constructor. We always try to fill the
        prompt to the max amount, the lowest similarity search will be cutoff at
        the numberOfToken line. Notice that we use the 3/4 token per character
        characteristic. We also assume that the last message in the conversation
        was the question. the System prompt is inserted just before the user
        question.

        This function returns the augmented conversation and a list of the used queries.
        """
        query = conversation.messages[-1]['content']
        n = self.initialNGuess
        while True:
            currentSearch = self.search.findNCClosest(query, n)
            if sum([len(result.text) for result in currentSearch])*0.75 > self.numberOfTokens:
                break
            n = n*2

        self.initialNGuess = max(floor(0.75 * n), 1) # If we have reached the desired number of prompts we can update our guess but we assume dat we overestimated a bit.
        context = 'The following information was found:\n'
        for result in currentSearch:
            context += f"From {result.reference.get} the following information was found:\n{result.text}"
        
        if self.resolver.supportsSystemRole:
            conversation.messages.insert(-1, {'role':'system', 'content': context})
        else:
            # Some resolvers that don't support the the role are very nitpick about alternating roles, therefore we add an additional assistant in this case.
            conversation.append_response('Can you give me some additional information first?')
            conversation.add_user_input(context)

        return conversation, currentSearch
        
                

