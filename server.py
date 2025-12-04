from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Init FastAPI app
app = FastAPI()

# Enable CORS (important for Hoppscotch / your frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # You can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Load Model
# ---------------------------
MODEL_NAME = "sshleifer/tiny-gpt2"   # Tiny model for Render stability

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

device = torch.device("cpu")   # Force CPU (Render has no GPU)
model.to(device)
model.eval()

# ---------------------------
# Request/Response Schemas
# ---------------------------
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    output: str

# ---------------------------
# Health Check
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------------------
# Generate Endpoint
# ---------------------------
@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            do_sample=True,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Remove prompt from beginning of output if model repeats it
    if full_text.startswith(req.prompt):
        full_text = full_text[len(req.prompt):].strip()

    return GenerateResponse(output=full_text)
