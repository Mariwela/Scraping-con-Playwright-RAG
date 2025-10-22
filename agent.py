import os
import re
import requests
from typing import Tuple

from rag import run_rag
from tools import newsapi_top_headlines, openweather_current


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")


def call_gemini_http(system: str, user: str, model: str = GOOGLE_MODEL, temperature: float = 0.2, max_tokens: int = 250) -> str:
    """Call Google Generative API (Gemini) via REST. Returns assistant text or raises on error.
    """
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set in environment")

    # Use the v1 generateContent endpoint and pass the API key in the header (x-goog-api-key)
    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": GOOGLE_API_KEY}
    # Build contents.parts similar to the curl example: include system and user as separate parts
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": system},
                    {"text": user},
                ]
            }
        ],

    }

    r = requests.post(url, json=payload, headers=headers, timeout=30)
    try:
        r.raise_for_status()
    except requests.HTTPError as he:
        code = r.status_code
        text = r.text
        if code == 401:
            raise RuntimeError("Google API unauthorized (401). Revisa tu GOOGLE_API_KEY.")
        if code == 402:
            raise RuntimeError("Google API billing issue (402). Es posible que hayas agotado la cuota o necesites actualizar tu plan.")
        if code == 429:
            raise RuntimeError("Google API rate limit (429). Intenta reducir la frecuencia o usar un modelo más pequeño.")
        raise RuntimeError(f"Google API error {code}: {text}")

    j = r.json()
    # Response shape can vary across versions. Try several fallbacks.
    try:
        # v1 generateContent may return 'candidates' with 'content' or 'output'
        if "candidates" in j and isinstance(j["candidates"], list) and j["candidates"]:
            cand = j["candidates"][0]
            content = cand.get("content") or cand.get("output") or cand.get("text")
            # content can be a list of parts, a string, or a dict with 'parts'
            if isinstance(content, dict):
                parts = content.get("parts") or content.get("content")
                if isinstance(parts, list):
                    texts = []
                    for part in parts:
                        if isinstance(part, dict) and "text" in part:
                            texts.append(part["text"])
                        elif isinstance(part, str):
                            texts.append(part)
                    if texts:
                        return "".join(texts).strip()
                # fallback: try to get 'text' directly
                if "text" in content and isinstance(content["text"], str):
                    return content["text"].strip()
            if isinstance(content, list):
                texts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        texts.append(part["text"])
                    elif isinstance(part, str):
                        texts.append(part)
                return "".join(texts).strip()
            if isinstance(content, str):
                return content.strip()

        # alternative: 'output' may contain 'content' list
        output = j.get("output")
        if isinstance(output, dict):
            cont = output.get("content") or output.get("parts")
            # cont might be a dict with 'parts'
            if isinstance(cont, dict):
                parts = cont.get("parts") or cont.get("content")
                if isinstance(parts, list):
                    texts = []
                    for item in parts:
                        if isinstance(item, dict) and "text" in item:
                            texts.append(item["text"])
                        elif isinstance(item, str):
                            texts.append(item)
                    if texts:
                        return "".join(texts).strip()
            if isinstance(cont, list):
                texts = []
                for item in cont:
                    if isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
                    elif isinstance(item, str):
                        texts.append(item)
                if texts:
                    return "".join(texts).strip()

        # final fallback: stringify entire response
        return str(j)
    except Exception:
        return str(j)


TOOL_PATTERN = re.compile(r"TOOL_CALL:\s*(?P<tool>\w+)\s*\|\s*(?P<param>.+)", re.IGNORECASE)


def process_tool_call(text: str):
    """Detecta una petición de tool en el texto y ejecuta la tool. Devuelve (tool_name, param, result) o None."""
    m = TOOL_PATTERN.search(text)
    if not m:
        return None
    tool = m.group("tool").strip()
    param = m.group("param").strip()
    if tool.lower() == "newsapi":
        res = newsapi_top_headlines(os.getenv("NEWSAPI_KEY"), param)
        return ("NewsAPI", param, res)
    if tool.lower() == "openweather":
        res = openweather_current(os.getenv("OPENWEATHER_KEY"), param)
        return ("OpenWeather", param, res)
    return (tool, param, {"success": False, "error": "Unknown tool"})


def answer_with_agent(user_query: str, collection, df) -> Tuple[str, list]:
    """High-level: run RAG to produce context, then use LLM to answer. The LLM can request tools using the special syntax:
    TOOL_CALL: ToolName|parameter

    The agent will execute at most 2 tool calls and ask the model to finish.
    Returns (final_text, history) where history contains intermediate tool results.
    """
    # 1) run rag to get summary and docs
    rag_summary = run_rag(user_query, collection, df)

    system = (
        "Eres un asistente experto que responde preguntas sobre medallas olímpicas. "
        "Puedes usar las herramientas NewsAPI y OpenWeather si necesitas información externa. "
        "Si decides ejecutar una herramienta, responde con una línea EXACTA con el formato:"
        "\nTOOL_CALL: ToolName|parameter\n" 
        "Por ejemplo: TOOL_CALL: NewsAPI|Argentina futbol"
        "\nSi no necesitas herramientas, entrega la respuesta final directamente."
    )

    user = f"Contexto RAG:\n{rag_summary}\n\nPregunta del usuario: {user_query}\n\nResponde o solicita TOOL_CALL si necesitas una herramienta."

    history = []

    try:
        assistant = call_gemini_http(system, user)
    except Exception as e:
        # Log the error for diagnostics and fallback to RAG summary
        print(f"⚠️ LLM call failed: {e}")
        return (f"(LLM no disponible) {rag_summary}", history)

    # check for tool call
    tool_call = process_tool_call(assistant)
    if not tool_call:
        return (assistant, history)

    # execute tool and provide result back to model
    tool_name, param, tool_res = tool_call
    history.append({"tool": tool_name, "param": param, "result": tool_res})

    tool_output_text = f"Tool {tool_name} output: {tool_res.get('result') if isinstance(tool_res, dict) else str(tool_res)}"

    # Ask model to finalize using tool output
    follow_up_user = (
        f"El resultado de la herramienta ({tool_name}) para '{param}' es:\n{tool_output_text}\n\n" 
        "Usa este resultado y el contexto RAG para dar la respuesta final al usuario."
    )
    try:
        final = call_gemini_http(system, follow_up_user + "\nPregunta original: " + user_query)
        return (final, history)
    except Exception as e:
        print(f"⚠️ LLM follow-up call failed: {e}")
        return (rag_summary, history)
