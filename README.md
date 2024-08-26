# llm-tools
This is a python script that sets up three services to support content tagging/search in the context of using LLM/RAG.

* extract keywords (in english, including synonyms, usable for tagging)
* extract nouns (in original language, usable to create search terms)
* summarization (in english)

The script is focused on privacy and resource efficiency. All models used can be downloaded locally, no data is sent to external services (like google translate). For better resource efficiency, a ModelPoolManager is applied to hold only the models for the languages that were accessed most recently.

