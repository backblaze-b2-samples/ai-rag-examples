# Retrieval-Augmented Generation (RAG) with Backblaze B2

This repo contains sample code showing how to pull context data from a [Backblaze B2 Cloud Object Storage](https://www.backblaze.com/cloud-storage) Bucket.

There are currently two Jupyter notebooks. Each one contains commentary and code for a different use case.

* [`gpt4all_demo.ipynb`](gpt4all_demo.ipynb) uses [GPT4All](https://www.nomic.ai/gpt4all) to load a large language model (LLM) and answer a series of related questions without any custom context.
* [`rag_demo.ipynb`](rag_demo.ipynb) uses the [LangChain framework](https://github.com/langchain-ai/langchain) to build a retrieval-augmented generation (RAG) chain that loads context from PDF data stored in a Backblaze B2 bucket and implements a basic chatbot.

[//]: # (* [`rag_history.ipynb`]&#40;rag_history.ipynb&#41; builds on `rag_demo.ipynb`, adding chat history to the RAG chain.)
                                                           
The webinar, [Leveraging your Cloud Storage Data in AI/ML Apps and Services](https://www.youtube.com/watch?v=WpOl1Y8IWhw), shows the two approaches in action.
