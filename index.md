# Large Language Models: Chatbots  voor  Domein  Specifieke  Vragen
- [Administration](./administration/): <br>
    Contains all the administrative documents.
    - [Invoices](./administration/invoices/):<br>
        Costs that were made, mainly for API access.
    - [Internship Documents](./administration/stagedocs/):<br>
        - [Internship Schedule](./administration/stagedocs/INTERNSHIP_SCHEDULE.doc)
        - [Risk Analysis](./administration/stagedocs/Risicoanalyse_nl.pdf)
        - [Work Station Sheet](./administration/stagedocs/werkpostfiche_nl%20.pdf)
- [Code](./code/):<br>
    The code that was used during the project.
    - [Original Code](./code/orignal_code/):<br>
        The code that was used to obtain the results of the paper.
        - [Chatbot Domain](./code/orignal_code/chatbot_domain/):<br>
            The source code of the module that was used during the BAP.
        - [Examination](./code/orignal_code/examination/):<br>
            The results of the manual testing that was done.
        - [OpenLLM Leader Board](./code/orignal_code/openllm%20leaderboard/):<br>
            The code that was used to scrape [HuggingFace's OpenLLM Leader Board](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
        - [Scores](./code/orignal_code/scores/):<br>
            The results that were obtained using the automated test bench.
        - [.env](./code/orignal_code/.env):<br>
            The API keys used during testing
        - [Questions](./code/orignal_code/questions.json):<br>
            The questions that were used for the automated test bench

    - [Improved Code](./code/improved_code/):<br>
        An improved version of the code that does not include a benchmarker. This version of the code was mainly used to have a nice finished product
        - [RAG Chatbot](./code/improved_code/rag_chatbot/):<br>
            The source code of the improved chatbot, includes a [README](./code/improved_code/rag_chatbot/README.md) that has a more detailed explanation of the code.
        - [Chatbot Example](./code/improved_code/chatbot_example/):<br>
            An example of how to use the improved code as a websocket implementation.
- [Paper](./paper/):<br>
    Contains the different versions of the paper.
    - [Final Paper](./paper/Bachelorproef_Finaal.pdf)
- [Presentation](./presentation/):<br>
    Contains the presentation used for the defense
- [Reports](./reports/): <br>
    Progress reports.

## Remote Sources:
- [HuggingFace Profile](https://huggingface.co/vincentverbergt):
  - [Finetuned Model](https://huggingface.co/vincentverbergt/Mistral7B-DIP)
  - [Finetuning Dataset](https://huggingface.co/datasets/vincentverbergt/DIP-sentences)
- [Github repository](https://github.com/centmeteenvin/6-bap)