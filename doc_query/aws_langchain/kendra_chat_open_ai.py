from doc_query.aws_langchain.kendra_index_retriever import KendraIndexRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All
from langchain.chat_models import ChatOpenAI
import sys
import os
from dotenv import load_dotenv

load_dotenv()

MAX_HISTORY_LENGTH = 5

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get("MODEL_PATH")
model_n_ctx = os.environ.get("MODEL_N_CTX")

def build_chain():
  region = os.environ["AWS_REGION"]

  kendra_index_id = os.environ["KENDRA_INDEX_ID"]

  # llm = ChatOpenAI(callbacks=[callback],streaming=True,verbose=True)
  match model_type:
    case "GPT4ALL":
      llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', verbose=False)
    case "OPENAI":
      llm = ChatOpenAI(model= "gpt-3.5-turbo",temperature=1, max_tokens=300,model_kwargs={'presence_penalty':0.86})
      
  retriever = KendraIndexRetriever(kendraindex=kendra_index_id, 
      awsregion=region, 
      return_source_documents=True)


  combine_docs_custom_prompt = PromptTemplate.from_template(
      ( '''The following is a conversation between a customer service executive in telecom services and an AI with knowledge about telecom from its context.
  The AI is descriptive and tries to provide lots of specific details from its context.
  If the AI does not know the answer to a question or the question lacks clarity, it truthfully says it 
  does not know and asks for futher details 
  for casual questions it answers from its own knowledge
  {context}
  Instruction: Based on the above documents, provide a detailed answer for, {question} Answer from knowledge if no answer is found in context Solution:
  ''')
)
  return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever,combine_docs_chain_kwargs=dict(prompt=combine_docs_custom_prompt),return_source_documents=True)
  

def run_chain(chain, prompt: str, history=[]):
  return chain({"question": prompt, "chat_history": history})


if __name__ == "__main__":
  class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

  qa = build_chain()
  chat_history = []
  print(bcolors.OKBLUE + "Hello! How can I help you?" + bcolors.ENDC)
  print(bcolors.OKCYAN + "Ask a question, start a New search: or CTRL-D to exit." + bcolors.ENDC)
  print(">", end=" ", flush=True)
  for query in sys.stdin:
    if (query.strip().lower().startswith("new search:")):
      query = query.strip().lower().replace("new search:","")
      chat_history = []
    elif (len(chat_history) == MAX_HISTORY_LENGTH):
      chat_history.pop(0)
    print(chat_history)
    result = run_chain(qa, query, chat_history)
    chat_history.append((query, result["answer"]))
    print(bcolors.OKGREEN + result['answer'] + bcolors.ENDC)
    if 'source_documents' in result:
      print(bcolors.OKGREEN + 'Sources:')
      for d in result['source_documents']:
        print(d.metadata['source'])
    print(bcolors.ENDC)
    print(bcolors.OKCYAN + "Ask a question, start a New search: or CTRL-D to exit." + bcolors.ENDC)
    print(">", end=" ", flush=True)
  print(bcolors.OKBLUE + "Bye" + bcolors.ENDC)
