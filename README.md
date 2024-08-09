# Retrieval-Augmented Generation (RAG) with Backblaze B2

Organizations looking to gain the benefits of AI, and, in particular, large language models (LLMs) must [guard against the risks of using public services such as OpenAI's ChatGPT](https://www.forbes.com/sites/kateoflahertyuk/2024/05/17/chatgpt-4o-is-wildly-capable-but-it-could-be-a-privacy-nightmare/). One solution is to run a private LLM, where you select a model, and can more safely provision it with private data as context for generating responses.   

This repository contains sample code showing how to build a retrieval-augmented generation (RAG) conversation chatbot application that loads context data in the form of PDFs from a private [Backblaze B2 Cloud Object Storage](https://www.backblaze.com/cloud-storage) Bucket.

There are two Jupyter notebooks:

* [`gpt4all_demo.ipynb`](gpt4all_demo.ipynb) uses [GPT4All](https://www.nomic.ai/gpt4all) to load a large language model (LLM) and answer a series of related questions without any custom context. This is a minimal example to show the basics of working with LLMs on your own machine.
* [`rag_demo.ipynb`](rag_demo.ipynb) uses the [LangChain framework](https://github.com/langchain-ai/langchain) to build a retrieval-augmented generation (RAG) chain that loads context from PDF data stored in a Backblaze B2 bucket and implements a conversational chatbot that can include message history in generating responses.

You can browse the notebooks on GitHub and see sample output, or [run them yourself](#running-the-notebooks).

The webinar, [Leveraging your Cloud Storage Data in AI/ML Apps and Services](https://www.youtube.com/watch?v=WpOl1Y8IWhw), shows the Python applications that on which the above notebooks are based:

[![Backblaze AI/ML Webinar on YouTube](https://github.com/user-attachments/assets/b87b22d0-aa63-469d-8276-b1d3d0a466e6)](https://www.youtube.com/watch?v=WpOl1Y8IWhw)

# Running the Notebooks

Both notebooks should run on any Jupyter-compatible platform:

* The "classic" Jupyter Notebook interface is lightweight and easy to install and use: https://docs.jupyter.org/en/latest/install/notebook-classic.html
* JupyterLab is more versatile, with more features: https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html
* There are Jupyter plugins for many integrated development environments (IDEs), for example, [IntelliJ](https://plugins.jetbrains.com/plugin/22814-jupyter) and [VS Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).

## JupyterLab Settings

If you are deploying JupyterLab on a virtual machine at a cloud provider, you will need to configure it to accept connections from the internet. Here is the configuration we set in `~/.jupyter/jupyter_server_config.py` for this purpose:

```python
# Allow requests where the Host header doesn't point to a local server
c.ServerApp.allow_remote_access = True

# The IP address the Jupyter server will listen on.
# 0.0.0.0 = all addresses
c.ServerApp.ip = '0.0.0.0'

# Allow access to hidden files
# Set to True to allow access to .venv
c.ContentsManager.allow_hidden = True
```
