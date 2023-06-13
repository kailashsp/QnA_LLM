from doc_query.aws_langchain.kendra_index_retriever import KendraIndexRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
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
  #model_kwargs={'presence_penalty':0.86}

  match model_type:
    case "GPT4ALL":
      llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', verbose=False)
    case "OPENAI":
      llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()],model= "gpt-3.5-turbo",temperature=0)
      
  retriever = KendraIndexRetriever(kendraindex=kendra_index_id, 
      awsregion=region, )
      # return_source_documents=True)

  response_schemas = [
    ResponseSchema(name="emotion", description="he emotion of the customer. It could be neutral/happy/sad/anger/frustration/annoyed/irritated/satisfaction."),
    ResponseSchema(name="suggestion", description='''suggestion means what is your suggestion to the Technician. 
                                                    It can be "continue to chat" if customer is happy/neutral. 
                                                    It could be "Escalate to live Agent" if customer's emotion is Anger/Annoyed/Irritated. 
                                                    It could be "Possible escalation" if emotion of customer is frustration/sadness.'''),
    ResponseSchema(name="response", description='''the value of "Response" should be output for the question which in our case is "What should be the next question asked by the technician?".''')                                
    ]
  output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
  format_instructions = output_parser.get_format_instructions()
  prompt = ChatPromptTemplate(
    messages=[HumanMessagePromptTemplate.from_template('''Analyse the following chat conversation {question}, understand the emotion of the customer from conversation and give response accordingly. 
                What should be the next question asked by the technician. Give output after searching through {context} for answers in the below format.
                {format_instructions}''')],
    input_variables=["question","context"],
    partial_variables={"format_instructions": format_instructions}
  )

  #     You are a support AI to customer support executive
  # - The questions will be from telecom field. 
  # - check whether the inputs are descriptive enough and from telecom industry
	# - The AI tries to answer from the {context} if there is a exact answer for {question} 
	# - Otherwise it will answer from its own knowledge and ask me questions until you have enough info
	# - Give me as many options which are your best guess as solution and answer as humanly as possible without repeating things from this prompt
#   combine_docs_custom_prompt = PromptTemplate.from_template('''""
# Analyse the chat conversation, {question} understand the emotion of the customer from conversation and give response accordingly. What should be the next question asked by the technician. Give output in the below format.

# "emotion": "  ",
# "suggestion":"  ",
# "response":"  "

# In the above format "emotion" means, the emotion of the customer. It could be neutral/happy/sad/anger/frustration/annoyed/irritated/satisfaction.
# The emotion that is sensed from customers input should be the value of "emotion". For example if you sense "anger"/"frustration" in customer's input, "emotion" should be "anger"/"frustration".
# In the above format "suggestion" means what is your suggestion to the Technician. It can be "continue to chat" if customer is happy/neutral. It could be "Escalate to live Agent" if customer's emotion is "Anger/Annoyed/Irritated". It could be "Possible escalation" if emotion of customer is "frustration/sadness".
# In the above format, the value of "Response" should be output for the question which in our case is "What should be the next question asked by the technician?".
# search for response from {context} as well
#   ''')



  return ConversationalRetrievalChain.from_llm(llm=llm,retriever=retriever,combine_docs_chain_kwargs=dict(prompt=prompt))
                                               #return_source_documents=True)
  

def run_chain(chain, prompt: str, history=[]):
  return chain({"question": prompt,"chat_history": history})


# if __name__ == "__main__":
#   class bcolors:
#     HEADER = '\033[95m'
#     OKBLUE = '\033[94m'
#     OKCYAN = '\033[96m'
#     OKGREEN = '\033[92m'
#     WARNING = '\033[93m'
#     FAIL = '\033[91m'
#     ENDC = '\033[0m'
#     BOLD = '\033[1m'
#     UNDERLINE = '\033[4m'

#   qa = build_chain()
#   chat_history = []
#   print(bcolors.OKBLUE + "Hello! How can I help you?" + bcolors.ENDC)
#   print(bcolors.OKCYAN + "Ask a question, start a New search: or CTRL-D to exit." + bcolors.ENDC)
#   print(">", end=" ", flush=True)
#   for query in sys.stdin:
#     if (query.strip().lower().startswith("new search:")):
#       query = query.strip().lower().replace("new search:","")
#       chat_history = []
#     elif (len(chat_history) == MAX_HISTORY_LENGTH):
#       chat_history.pop(0)
#     print(chat_history)
#     result = run_chain(qa, query, chat_history)
#     chat_history.append((query, result["answer"]))
#     print(bcolors.OKGREEN + result['answer'] + bcolors.ENDC)
#     if 'source_documents' in result:
#       print(bcolors.OKGREEN + 'Sources:')
#       for d in result['source_documents']:
#         print(d.metadata['source'])
#     print(bcolors.ENDC)
#     print(bcolors.OKCYAN + "Ask a question, start a New search: or CTRL-D to exit." + bcolors.ENDC)
#     print(">", end=" ", flush=True)
#   print(bcolors.OKBLUE + "Bye" + bcolors.ENDC)
