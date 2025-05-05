# app.py
from fastapi import FastAPI, Request, HTTPException
from starlette.responses import JSONResponse

# Import your existing functions directly
from graphrag_sakir import generate_ideal
from proactive_chain_of_thought_gaurdrails import evaluate_answer_by_rubric_items

app = FastAPI()

@app.post("/generate_ideal")
async def gen_ideal(req: Request):
    payload = await req.json()
    try:
        # assume payload contains {"text": "..."}
        return JSONResponse(generate_ideal(payload["text"]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate")
async def evaluate(req: Request):
    payload = await req.json()
    try:
        # assume payload has {"question": "...", "student_answer": "..."}
        score, history = evaluate_answer_by_rubric_items(
            payload["question"], payload["student_answer"]
        )
        return JSONResponse({"score": score, "history": history})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/soften_score")
async def soften(req: Request):
    payload = await req.json()
    try:
        # assume payload {"score": 7.5}
        return JSONResponse({"softened_score": soften_score(payload["score"])})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
