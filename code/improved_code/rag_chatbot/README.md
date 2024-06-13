### Rag chatbot
This chatbot package has the intention of easily and modularity create a pipeline for a RAG chatbot.

## Pipeline
A pipeline consists of the following parts
1. A Text source
    - This can be a webpage, a pdf document.
2. Text cleanup
   - This can be some Regex or filters to remove artefact from the source.
3. A Similarity search
   - This step might contain some pre-computation or might happen realtime
   - This might be vector retrieval or n-gram-Similarity
4. A language model
   - This can take the question and the RAG context and convert it to a clean answer

That is it. That is the pipeline.

## Caching
Some operation may take a considered amount of time and are therefore cached in te background. This process is a transparent as we can possibly make it but may take up a considerable amount of storage. Therefore you can flush the entire command by using the following command: 
<br>`python -m rag_chatbot.clear_cache`