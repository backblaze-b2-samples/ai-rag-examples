# Retrieval-Augmented Generation (RAG) with Backblaze B2

This repo contains sample code showing how to build a retrieval-augmented generation (RAG) conversation chatbot application that loads context data in the form of PDFs from a [Backblaze B2 Cloud Object Storage](https://www.backblaze.com/cloud-storage) Bucket and implements message history.

There are two Jupyter notebooks:

* [`gpt4all_demo.ipynb`](gpt4all_demo.ipynb) uses [GPT4All](https://www.nomic.ai/gpt4all) to load a large language model (LLM) and answer a series of related questions without any custom context.
* [`rag_demo.ipynb`](rag_demo.ipynb) uses the [LangChain framework](https://github.com/langchain-ai/langchain) to build a retrieval-augmented generation (RAG) chain that loads context from PDF data stored in a Backblaze B2 bucket and implements a conversational chatbot.

The webinar, [Leveraging your Cloud Storage Data in AI/ML Apps and Services](https://www.youtube.com/watch?v=WpOl1Y8IWhw), shows the two approaches in action:

[![Backblaze AI/ML Webinar on YouTube](https://github.com/user-attachments/assets/b87b22d0-aa63-469d-8276-b1d3d0a466e6)](https://www.youtube.com/watch?v=WpOl1Y8IWhw)
