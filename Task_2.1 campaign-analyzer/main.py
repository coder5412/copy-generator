from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from openai import OpenAI

app = FastAPI(title="AI Campaign Brief Analyzer")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Request Schema
class BriefRequest(BaseModel):
    brief_text: str

# Response Schema
class BriefResponse(BaseModel):
    audience: str
    key_messages: List[str]
    tone: str
    channels: List[str]
    risks: List[str]


@app.post("/analyze-brief", response_model=BriefResponse)
async def analyze_brief(request: BriefRequest):

    if len(request.brief_text) < 50:
        raise HTTPException(status_code=400, detail="Brief too short")

    prompt = f"""
You are a senior marketing strategist.

Analyze the campaign brief below and extract:
- audience
- key_messages (list)
- tone
- channels (list)
- risks (list)

Return ONLY valid JSON in this format:
{{
  "audience": "...",
  "key_messages": ["..."],
  "tone": "...",
  "channels": ["..."],
  "risks": ["..."]
}}

Campaign Brief:
{request.brief_text}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        output = response.choices[0].message.content

        return eval(output)  # simple parsing (for demo; production: use json.loads)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))