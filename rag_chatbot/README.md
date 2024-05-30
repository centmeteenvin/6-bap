### Rag chatbot
This chatbot package has the intention of easily and modularity create a pipeline for a RAG chatbot.

## Pipeline
A pipeline consists of the following parts
1. A Text source
    - This can be a webpage, a pdf document.
2. Text cleanup
   - This can be some Regex or filters to remove artefact from the source.
3. A Similarity search
   - This step might contain some precomputation or might happen realtime
   - This might be vector retrieval or ngram-Similarity
4. A language model
   - This can take the question and the RAG context and convert it to a clean answer

That is it. That is the pipeline.