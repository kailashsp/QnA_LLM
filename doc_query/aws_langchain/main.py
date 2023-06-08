from fastapi import FastAPI, Request
from kendra_chat_open_ai import build_chain,run_chain
app = FastAPI()
qa = build_chain()


@app.post("/answer")
async def answer_question(query:str ):
    data = query
    print(data)
    prompt = data
    result = run_chain(qa, prompt)
    return {"answer": result["answer"],"source_documents": result.get("source_documents", [])}


