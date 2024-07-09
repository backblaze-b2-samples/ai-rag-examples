import atexit
import os
import readline

from gpt4all import GPT4All

from dotenv import load_dotenv

if load_dotenv():
    print('Loaded environment variables from .env')
else:
    print('No environment variables in .env!')

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


def main():
    model = GPT4All(
        model_name=model_name,
        model_path=model_directory,
        device=device,
        n_ctx=int(max_context_window)
    )

    prompt_template = """Answer the question below.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
        ### Question: {0}
        
        ### Response:
        """

    with model.chat_session(
        prompt_template=prompt_template
    ):
        while True:
            question = input('\nAsk me a question: ')
            if question and len(question) > 0:
                answer = model.generate(question)
                print(answer)


if __name__ == '__main__':
    main()
