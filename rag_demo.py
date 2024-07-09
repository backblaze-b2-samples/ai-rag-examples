import atexit
import os
# noinspection PyUnresolvedReferences
import readline
from datetime import datetime
from shutil import copyfileobj
from zipfile import ZipFile, ZipInfo

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import S3FileLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import GPT4All
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from s3fs import S3FileSystem

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

PDF_LOCATION = 'pdfs'

VECTOR_DB_ARCHIVE = 'vectordb/vectordb.zip'
VECTOR_DB_DIRECTORY = 'vectordb'
VECTOR_DB_FILE = 'chroma.sqlite3'

if load_dotenv():
    print('Loaded environment variables from .env')
else:
    print('No environment variables in .env!')

bucket_name = os.environ['BUCKET_NAME']
model_name = os.environ['MODEL_NAME']
model_directory = os.environ['MODEL_DIRECTORY']
max_context_window = os.environ['MAX_CONTEXT_WINDOW']
device = os.environ.get('DEVICE', 'auto')

history_file = ".rag-history"


def save_history():
    readline.write_history_file(history_file)


if os.path.exists(history_file):
    readline.read_history_file(history_file)
atexit.register(save_history)

b2fs = S3FileSystem(version_aware=True)


def create_vector_database_from_docs():
    print(f'Loading PDF data from B2 bucket {bucket_name}/{PDF_LOCATION}')
    data = []
    for file in b2fs.glob(f'{bucket_name}/{PDF_LOCATION}/*.pdf'):
        filename = file.removeprefix(f'{bucket_name}/')
        print(f'Loading {filename}')
        loader = S3FileLoader(bucket_name, filename)
        data += loader.load()
        print(f'Loaded {len(data)} document(s)')

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_splits = text_splitter.split_documents(data)
    print(f'Split doc(s) into {len(all_splits)} chunks')

    chroma = Chroma.from_documents(documents=all_splits,
                                   embedding=GPT4AllEmbeddings(model_name='all-MiniLM-L6-v2.gguf2.f16.gguf'),
                                   persist_directory=VECTOR_DB_DIRECTORY)
    print('Created vector store')

    return chroma


def upload_vector_database_to_b2():
    with b2fs.open(f'{bucket_name}/{VECTOR_DB_ARCHIVE}', mode='wb') as f, ZipFile(f, mode='w') as zipfile:
        for root, _dirnames, filenames in os.walk(VECTOR_DB_DIRECTORY):
            for filename in filenames:
                fullpath = os.path.join(root, filename)
                mtime = os.path.getmtime(fullpath)
                last_modified = datetime.fromtimestamp(mtime)
                date_time = (last_modified.year, last_modified.month, last_modified.day,
                             last_modified.hour, last_modified.minute, last_modified.second)
                # Want path relative to VECTOR_DB_DIRECTORY
                zipinfo = ZipInfo(filename=fullpath.removeprefix(f'{VECTOR_DB_DIRECTORY}/'), date_time=date_time)
                with open(fullpath, mode='rb') as src, zipfile.open(zipinfo, mode='w') as dst:
                    copyfileobj(src, dst)
    print('Uploaded vector store')


def download_vector_database_from_b2():
    if not os.path.isdir(VECTOR_DB_DIRECTORY):
        os.mkdir(VECTOR_DB_DIRECTORY)
    with b2fs.open(f'{bucket_name}/{VECTOR_DB_ARCHIVE}', mode='rb') as f, ZipFile(f, mode='r') as myzip:
        myzip.extractall(VECTOR_DB_DIRECTORY)
    print('Downloaded and extracted vector store from B2')


def load_local_vector_database():
    chroma = Chroma(embedding_function=GPT4AllEmbeddings(model_name='all-MiniLM-L6-v2.gguf2.f16.gguf'),
                    persist_directory="./vectordb")
    # noinspection PyProtectedMember
    print(f'Loaded vector store from local disk. {chroma._collection.count()} embeddings.')
    return chroma


def get_vector_store():
    """
    This function uses the following logic to instantiate a Chroma vector database:
    * If there is a local vector database at `vectordb/chroma.sqlite3`, use it, otherwise,
    * If there is an archived vector database at `my-bucket/vectordb.zip`, download it and use it, otherwise,
    * Create a local vector database from the PDFs in B2, save it locally, archive it to B2, and use it.
    """
    if os.path.exists(f'{VECTOR_DB_DIRECTORY}/{VECTOR_DB_FILE}'):
        print('Loading vector store from local disk')
        vectorstore = load_local_vector_database()
    else:
        try:
            print('Looking for vector store in B2')
            download_vector_database_from_b2()
            vectorstore = load_local_vector_database()
        except FileNotFoundError:
            print('No vector store in B2; creating vector store from docs and uploading to B2')
            vectorstore = create_vector_database_from_docs()
            upload_vector_database_to_b2()
    return vectorstore


def main():
    # If there is a local vector database, use it, otherwise,
    # If there is an archived vector database in B2, download it and use it, otherwise,
    # Create a local vector database from the PDFs in B2, save it locally, archive it to B2, and use it.
    vectorstore = get_vector_store()

    # Sanity check on the vector store
    question = "When would you use a master application key?"
    search_results = vectorstore.similarity_search(question)
    print(f'Test similarity search: found {len(search_results)} docs')
    print(f'First doc ({len(search_results[0].page_content)} characters): {search_results[0]}')

    # Load an LLM
    print(f'Loading LLM, requesting device {device}')
    model = GPT4All(
        model=os.path.join(model_directory, model_name),
        max_tokens=int(max_context_window),
        device=device
    )
    print(f'Loaded LLM, running on {model.device}')

    # This is the example Q&A RAQ prompt from
    # https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/retrieval_qa/prompt.py
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    {context}
    
    Question: {question}
    Helpful Answer:"""
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # The retriever knows how to take string queries and return the most 'relevant' Documents from its source.
    retriever = vectorstore.as_retriever()

    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )

    while True:
        question = input('\nAsk me a question about Backblaze B2: ')
        if question and len(question) > 0:
            answer = chain.invoke(question)
            print(answer)


if __name__ == '__main__':
    main()
