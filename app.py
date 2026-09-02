from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re


# =========================================================
# 1. FASTAPI APP
# =========================================================

app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0"
)


# =========================================================
# 2. MODEL & TOKENIZER
# =========================================================

MODEL_PATH = "./saved_summary_model"

print("Loading model...")

tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

print("Model loaded successfully!")


# =========================================================
# 3. DEVICE
# =========================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model.to(device)
model.eval()


# =========================================================
# 4. HTML TEMPLATE
# =========================================================

templates = Jinja2Templates(directory=".")


# =========================================================
# 5. INPUT SCHEMA
# =========================================================

class DialogueInput(BaseModel):
    dialogue: str


# =========================================================
# 6. CLEAN TEXT
# =========================================================

def clean_data(text: str):

    # Remove line breaks
    text = re.sub(r"\r\n|\r|\n", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


# =========================================================
# 7. SUMMARIZATION FUNCTION
# =========================================================

def summarize_dialogue(dialogue: str):

    dialogue = clean_data(dialogue)

    if not dialogue:
        return "Please enter some text to summarize."

    # Tokenize input
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    # Move tensors to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generate summary
    with torch.no_grad():

        targets = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    # Decode output
    summary = tokenizer.decode(
        targets[0],
        skip_special_tokens=True
    )

    return summary


# =========================================================
# 8. SUMMARIZE API
# =========================================================

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):

    summary = summarize_dialogue(
        dialogue_input.dialogue
    )

    return {
        "summary": summary
    }


# =========================================================
# 9. HOME PAGE
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request
        }
    )