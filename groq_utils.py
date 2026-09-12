"""
groq_utils.py — Groq client setup and error-insertion helper.
Imported by notebooks; edit this file to change any logic.
"""

import os
import requests
from groq import Groq
from dotenv import load_dotenv, find_dotenv

from config import PREFERRED_MODELS, ERROR_TYPES, TEMPERATURE, MAX_TOKENS


# ── Prompt template ────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
You are a dataset-annotation assistant. Your task is to insert exactly ONE factual error \
into the RESPONSE below. Use the reference DOCUMENTS and QUESTION only to understand context \
— do NOT alter them.

Choose ONE error type at random from this list:
  Temporal, Numerical, Entity, Relation, Contradictory, Unverifiable

Tagging rules (follow exactly, no exceptions):
  • Span-level errors (Temporal / Numerical / Entity / Relation):
      <TYPE><delete>original_span</delete><mark>corrupted_span</mark></TYPE>
    Replace TYPE with the chosen error type in the exact casing shown above.
  • Sentence-level errors (Contradictory / Unverifiable):
      <TYPE>corrupted_sentence</TYPE>
    The corrupted sentence must replace the original sentence entirely inside the tag.

Constraints:
  - Insert EXACTLY one error. No more.
  - Output ONLY the tagged corrupted response. No explanation, no preamble.
  - Do NOT change any part of the response that is not involved in the error.
  - Do NOT add new sentences. Only corrupt or replace an existing span or sentence.

---
DOCUMENTS:
{documents}

QUESTION:
{question}

RESPONSE:
{response}
---

Output the tagged corrupted response now:"""


def load_api_key():
    """Load GROQ_API_KEY from .env (searches cwd and all parent dirs)."""
    env_file = find_dotenv(usecwd=True)
    load_dotenv(env_file, override=True)
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError("GROQ_API_KEY not found. Add it to FRED/.env")
    print(f".env  : {env_file}")
    return key


def pick_model(api_key):
    """Query Groq /models and return the first PREFERRED_MODELS match."""
    resp = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    available = {m["id"] for m in resp.json().get("data", [])}

    model = next((m for m in PREFERRED_MODELS if m in available), None)
    if not model:
        # fallback: any non-audio/guard model
        text_models = sorted(
            m for m in available
            if not any(x in m for x in ["whisper", "guard", "orpheus"])
        )
        model = text_models[0] if text_models else None

    if not model:
        raise ValueError(f"No usable model found. Available: {available}")

    print(f"Model : {model}")
    return model


def make_client(api_key):
    return Groq(api_key=api_key)


def format_documents(docs):
    if isinstance(docs, list):
        return "\n\n".join(f"[Doc {i+1}] {d}" for i, d in enumerate(docs))
    return str(docs)


def insert_error(client, model, documents_raw, question, response):
    """Call Groq to insert one tagged error into the response."""
    prompt = PROMPT_TEMPLATE.format(
        documents=format_documents(documents_raw),
        question=question,
        response=response,
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return completion.choices[0].message.content.strip()
