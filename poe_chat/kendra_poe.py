import os
import poe
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from poe_chat.index_retriever import KendraIndexRetriever
from poe_chat.output_template import EMOTION,FURTHER_QUESTIONS,SOLUTIONS,SUGGESTION

load_dotenv()

def poechat(convo):
    token = os.environ["POE_TOKEN"]
    index = os.environ["KENDRA_INDEX_ID"]
    region = os.environ["AWS_REGION"]


    client = poe.Client(token)
    response_schemas = [
    ResponseSchema(name="emotion", description=EMOTION),
    ResponseSchema(name="suggestion", description=SUGGESTION),
    ResponseSchema(name="solutions", description=SOLUTIONS)  ,
    ResponseSchema(name="further_questions",description=FURTHER_QUESTIONS)                              
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    system_instructions = open("poe_chat/prompts.txt","r").read()

    docs = KendraIndexRetriever(index,region)

    contents = docs.get_relevant_documents(query="what is ONT?")
    content =contents[0].page_content
    context = content.split('Document Excerpt:')[1]

    Prompt = system_instructions.format(question=convo,format_instructions=format_instructions,context=context)
    result = []
    client.send_chat_break("a2")
    for chunk in client.send_message("a2", Prompt):
        result.append(chunk["text_new"])
    
    result = "".join(result)
    res = result.split('```json\n')[1].split('\n```')[0]
    return res
    # for chunk in client.send_message("a2", message=Prompt):
    #     print(chunk)
    #     return chunk["text_new"]