import os
import gradio as gr
from vector_db import create_vector_db, query_vector_db
from scraper import scrape_medal_table
from rag import run_rag
from agent import answer_with_agent
import os


def setup():
    df = scrape_medal_table()
    collection, df_clean = create_vector_db(df)
    return collection, df_clean


collection, df_clean = setup()


def answer(query: str):
    # If GOOGLE_API_KEY is present, use the Gemini agent which may call tools.
    if os.getenv("GOOGLE_API_KEY"):
        final, history = answer_with_agent(query, collection, df_clean)
        hist_text = "\n\n".join([f"{h['tool']}({h['param']}): {h['result']}" for h in history])
        return final, hist_text

    # fallback: only run RAG
    rag_answer = run_rag(query, collection, df_clean)
    return rag_answer, "(No GOOGLE_API_KEY set; tools no disponibles)"


with gr.Blocks() as demo:
    gr.Markdown("# RAG + Tools demo")
    with gr.Row():
        query_in = gr.Textbox(label="Consulta")
    out_rag = gr.Textbox(label="RAG respuesta", lines=12)
    out_tool = gr.Textbox(label="Tool output")
    btn = gr.Button("Enviar")
    btn.click(answer, inputs=[query_in], outputs=[out_rag, out_tool])


def main():
    demo.launch()


if __name__ == "__main__":
    main()
