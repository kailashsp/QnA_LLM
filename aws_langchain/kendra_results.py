from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
import boto3
import re

def clean_result(res_text):
    res = re.sub("\s+", " ", res_text).replace("...","")
    return res
    
def get_top_n_results(resp, count):
    r = resp["ResultItems"][count]
    doc_title = r["DocumentTitle"]["Text"]
    doc_uri = r["DocumentURI"]
    r_type = r["Type"]
    if (r["AdditionalAttributes"] and r["AdditionalAttributes"][0]["Key"] == "AnswerText"):
        res_text = r["AdditionalAttributes"][0]["Value"]["TextWithHighlightsValue"]["Text"]
    else:
        res_text = r["DocumentExcerpt"]["Text"]
    doc_excerpt = clean_result(res_text)
    combined_text = "Document Title: " + doc_title + "\nDocument Excerpt: \n" + doc_excerpt + "\n"
    return {"page_content":combined_text, "metadata":{"source":doc_uri, "title": doc_title, "excerpt": doc_excerpt, "type": r_type}}

def kendra_query(kclient, kquery, kcount, kindex_id):
    print(type(kquery))
    prompt = PromptTemplate(
    input_variables=["convo"],
    template="Summarise the conversation {convo}?",
)

    llm= OpenAI(temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)
    kquery = chain.run(kquery)

    response = kclient.query(IndexId=kindex_id, QueryText=kquery.strip())
    if len(response["ResultItems"]) > kcount:
        r_count = kcount
    else:
        r_count = len(response["ResultItems"])
    docs = [get_top_n_results(response, i) for i in range(0, r_count)]
    return [Document(page_content = d["page_content"], metadata = d["metadata"]) for d in docs]

def kendra_client(kindex_id, kregion):
    kclient = boto3.client('kendra', region_name=kregion)
    return kclient
