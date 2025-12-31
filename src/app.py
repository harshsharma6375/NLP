import sys
import os
import json
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# Ensure src directory is in python path to allow imports from sibling modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline import analyze_conversation

app = FastAPI(title="NLP Conversation Analyzer")

# Setup templates
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates_dir = os.path.join(base_dir, "templates")
templates = Jinja2Templates(directory=templates_dir)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/analyze", response_class=HTMLResponse) # Redirect GET to root/form
async def analyze_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_text(request: Request, text: str = Form(...)):
    if not text or not text.strip():
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Please enter a conversation transcript.",
            "input_text": text
        })
    
    try:
        # Run the NLP pipeline
        analysis_result = analyze_conversation(text)
        
        # Format JSON for display
        formatted_result = json.dumps(analysis_result, indent=4)
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": formatted_result,
            "input_text": text
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"An error occurred during analysis: {str(e)}",
            "input_text": text
        })

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
