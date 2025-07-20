# app.py
from fastapi import FastAPI, Request, HTTPException
from starlette.responses import JSONResponse

# Import your existing functions directly
from graphrag_sakir import rag_generate
from proactive_chain_of_thought_gaurdrails import evaluate_answer_by_rubric_items
from softner import predict_softened_score

app = FastAPI()

@app.post("/generate_ideal")
async def gen_ideal(req: Request):
    payload = await req.json()
    try:
        # assume payload contains {"question": "...", "rubric_items": [...]}
        return JSONResponse(rag_generate(payload["question"], payload["rubric_items"]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate")
async def evaluate(req: Request):
    payload = await req.json()
    try:
        # assume payload has {"question": "...", "student_answer": "...", "ideal_answer": "...", "rubric_items": [...]}
        evaluation = evaluate_answer_by_rubric_items(
            payload["question"], payload["ideal_answer"], payload["student_answer"], payload["rubric_items"]
        )
        return JSONResponse(evaluation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/soften_score")
async def soften(req: Request):
    payload = await req.json()
    try:
        # assume payload {"procot_score": 7.5, "student_answer": "...", "ideal_answer": "..."}
        softened_score = predict_softened_score(
            payload["procot_score"], payload["student_answer"], payload["ideal_answer"]
        )
        return JSONResponse({"softened_score": softened_score})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
