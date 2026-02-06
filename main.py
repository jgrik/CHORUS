from fastapi import FastAPI
from pydantic import BaseModel
import datetime
import anthropic
import openai
import together
import os
from dotenv import load_dotenv
from database import init_database, store_results
load_dotenv()

app = FastAPI(title="Chorus API", version="0.2.0")
init_database()

claude_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)
openai_client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
together_client = together.Together(
    api_key=os.getenv("TOGETHERAI_API_KEY")
)

# Shared Configuration
SAFETY_SYSTEM_PROMPT = """You are a content safety classifier.

Analyze the user's prompt and determine if it's requesting harmful, illegal, or unsafe content.

IMPORTANT: Start your response with EXACTLY one of these:
VERDICT: SAFE
VERDICT: UNSAFE

Then explain your reasoning in 2-3 sentences.

Example safe response:
VERDICT: SAFE
This prompt asks for educational information about photosynthesis, which is harmless scientific content.

Example unsafe response:
VERDICT: UNSAFE
This prompt requests instructions for creating dangerous weapons, which could cause harm."""

# Request/Response Models
class AnalysisRequest(BaseModel):
    content: str

class ModelResponse(BaseModel):
    model_name: str
    safe: bool
    concerns: list
    reasoning: str

class AnalysisResponse(BaseModel):
    content: str
    claude_response: ModelResponse
    timestamp: str


# Helper Functions
def extract_verdict(response_text: str) -> str:
    """
    Extract safety verdict from first line of response.
    Only checks first line to prevent false positives.
    
    Returns: 'safe', 'unsafe', or 'unknown'
    """
    first_line = response_text.strip().split('\n')[0].upper()
    
    if 'VERDICT: SAFE' in first_line:
        return 'safe'
    elif 'VERDICT: UNSAFE' in first_line:
        return 'unsafe'
    else:
        return 'unknown'

def evaluate_consensus(claude_result: dict , gpt5_result: dict, llama_result: dict) -> dict:
    """
    Takes all model responses, returns consensus verdict.
    """
    safe_votes = sum([
        claude_result['safe'],
        gpt5_result['safe'],
        llama_result['safe']])
    if safe_votes == 3:
        return {
            'verdict':'SAFE',
            'confidence':'high',
            'reasoning':'All three models flagged this output as safe.',
            'flagged_by': []
        }
    if safe_votes == 0:
        return {
            'verdict':'UNSAFE',
            'confidence':'high',
            'reasoning':'All three models flagged this output as unsafe',
            'flagged_by': ['Claude' , 'GPT-5' , 'Llama']
        }
    flagged_by = []
    if not claude_result['safe']:
        flagged_by.append('Claude')
    if not gpt5_result['safe']:
        flagged_by.append('GPT-5')
    if not llama_result['safe']:
        flagged_by.append('Llama')
    return {
        'verdict' : 'REVIEW_REQUIRED',
        'confidence' : 'uncertain',
        'reasoning' : f'{len(flagged_by)} of 3 models flagged concerns',
        'flagged_by' : flagged_by
    }

async def analyze_with_claude(content: str) -> ModelResponse:
    """
    Analyzes content using Claude with structured VERDICT format.
    Returns standardized safety assessment.
    """
    
    system_prompt = SAFETY_SYSTEM_PROMPT

    try:
        message = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": content}
            ]
        )
        
        response_text = message.content[0].text
        
    except Exception as e:
        return ModelResponse(
            model_name="Claude Sonnet 4",
            safe=False,
            concerns=[f'API Error: {str(e)}'],
            reasoning="Error occurred during analysis"
        )
    
    verdict = extract_verdict(response_text)
    
    concerns = []
    if verdict == 'unsafe':
        reasoning_lines = response_text.split('\n')[1:]
        reasoning = ' '.join(reasoning_lines).strip()
        if reasoning:
            concerns.append(reasoning[:200])
    
    return ModelResponse(
        model_name="Claude Sonnet 4",
        safe=verdict == 'safe',
        concerns=concerns,
        reasoning=response_text[:200]
    )


async def analyze_with_gpt5(content: str) -> ModelResponse:
    """
    Analyzes content using GPT-5.2 Thinking with structured VERDICT format.
    Returns standardized safety assessment.
    """
    
    system_prompt = SAFETY_SYSTEM_PROMPT

    try:
        response = openai_client.chat.completions.create(
            model="gpt-5.2",
            max_completion_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]
        )
        
        response_text = response.choices[0].message.content
        
    except Exception as e:
        return ModelResponse(
            model_name="GPT 5.2 Thinking",
            safe=False,
            concerns=[f'API Error: {str(e)}'],
            reasoning="Error occurred during analysis"
        )
    
    verdict = extract_verdict(response_text)
    
    concerns = []
    if verdict == 'unsafe':
        reasoning_lines = response_text.split('\n')[1:]
        reasoning = ' '.join(reasoning_lines).strip()
        if reasoning:
            concerns.append(reasoning[:200])
    
    return ModelResponse(
        model_name="GPT 5.2 Thinking",
        safe=verdict == 'safe',
        concerns=concerns,
        reasoning=response_text[:200]
    )


async def analyze_with_llama(content: str) -> ModelResponse:
    """
    Analyzes content using Llama 3.1 70B with structured VERDICT format.
    Returns standardized safety assessment.
    """
    
    system_prompt = SAFETY_SYSTEM_PROMPT

    try:
        response = together_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]
        )
        
        response_text = response.choices[0].message.content
        
    except Exception as e:
        return ModelResponse(
            model_name="Llama 3.1 70B",
            safe=False,
            concerns=[f'API Error: {str(e)}'],
            reasoning="Error occurred during analysis"
        )
    
    verdict = extract_verdict(response_text)
    
    concerns = []
    if verdict == 'unsafe':
        reasoning_lines = response_text.split('\n')[1:]
        reasoning = ' '.join(reasoning_lines).strip()
        if reasoning:
            concerns.append(reasoning[:200])
    
    return ModelResponse(
        model_name="Llama 3.1 70B",
        safe=verdict == 'safe',
        concerns=concerns,
        reasoning=response_text[:200]
    )


# API Endpoints
@app.get("/")
async def root():
    return {
        "name": "Chorus API",
        "version": "0.2.0",
        "status": "running",
        "description": "Multi-model AI safety verification",
        "models": ["Claude Sonnet 4" , "GPT 5.2 Thinking" , "Llama 3.1 70B"]
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/analyze-claude")
async def analyze_content(request: AnalysisRequest):
    claude_response = await analyze_with_claude(request.content)
    
    return AnalysisResponse(
        content=request.content,
        claude_response=claude_response,
        timestamp=datetime.datetime.now().isoformat()
    )

@app.post("/analyze-gpt5")
async def analyze_gpt5_content(request: AnalysisRequest):
    gpt5_response = await analyze_with_gpt5(request.content)

    return {
        "content": request.content,
        "gpt5_response": gpt5_response.dict(),
        "timestamp": datetime.datetime.now(). isoformat()
    }

@app.post("/analyze-llama")
async def analyze_llama_content(request: AnalysisRequest):
    llama_response = await analyze_with_llama(request.content)

    return {
        "content": request.content,
        "llama_response": llama_response.dict(),
        "timestamp": datetime.datetime.now(). isoformat()
    }

@app.post("/analyze-all")
async def analyze_all_models(request: AnalysisRequest):
    """
    Analyzes content with all three models and returns consensus verdict.
    """
    
    claude_response = await analyze_with_claude(request.content)
    gpt5_response = await analyze_with_gpt5(request.content)
    llama_response = await analyze_with_llama(request.content)

    claude_dict = claude_response.dict()
    gpt5_dict = gpt5_response.dict()
    llama_dict = llama_response.dict()

    consensus = evaluate_consensus(claude_dict, gpt5_dict, llama_dict)

    store_results(
        request.content,
        claude_dict,
        gpt5_dict,
        llama_dict,
        consensus
    )

    return {
        "content" : request.content,
        "claude" : claude_dict,
        "gpt5" : gpt5_dict,
        "llama" : llama_dict,
        "consensus" : consensus,
        "timestamp" : datetime.datetime.now().isoformat()
    }
