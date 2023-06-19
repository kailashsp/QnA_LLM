import os
import sys
from aws_langchain.output_template import EMOTION,SUGGESTION,SOLUTIONS,FURTHER_QUESTIONS
from aws_langchain.kendra_index_retriever import KendraIndexRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All ,OpenAI, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from dotenv import load_dotenv

load_dotenv()

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get("MODEL_PATH")
model_n_ctx = os.environ.get("MODEL_N_CTX")
region = os.environ["AWS_REGION"]
kendra_index_id = os.environ["KENDRA_INDEX_ID"]

def build_chain():


  match model_type:
    case "LlamaCpp":
        llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, verbose=False)
    case "GPT4ALL":
      llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', verbose=False)
    case "OPENAI":
      #streaming to return the token one by one as it is generated
      #temperature is 0 to generate consistent results for same questions
      llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()],model= "gpt-3.5-turbo",temperature=0)
      
  retriever = KendraIndexRetriever(kendraindex=kendra_index_id, awsregion=region)
 

  response_schemas = [
    ResponseSchema(name="emotion", description=EMOTION),
    ResponseSchema(name="suggestion", description=SUGGESTION),
    ResponseSchema(name="solutions", description=SOLUTIONS)  ,
    ResponseSchema(name="further_questions",description=FURTHER_QUESTIONS)                              
    ]
  output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
  format_instructions = output_parser.get_format_instructions()
  system_instructions = open("aws_langchain/prompts.txt","r").read()
  prompt = ChatPromptTemplate(
    messages=[HumanMessagePromptTemplate.from_template(system_instructions)],
                                                          input_variables=["question","context"],
                                                          partial_variables={"format_instructions": format_instructions})

  return ConversationalRetrievalChain.from_llm(llm=llm,retriever=retriever,combine_docs_chain_kwargs=dict(prompt=prompt))


def run_chain(chain, prompt: str, history=[]):
  return chain({"question": prompt,"chat_history": history})
