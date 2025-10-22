import os
import gradio as gr
from vector_db import create_vector_db, query_vector_db
from scraper import scrape_medal_table
from rag import run_rag
from tools import newsapi_top_headlines, openweather_current


def setup():
    df = scrape_medal_table()
    collection, df_clean = create_vector_db(df)
    return collection, df_clean


collection, df_clean = setup()


def answer(query: str, tool: str, tool_input: str):
    # run rag
    rag_answer = run_rag(query, collection, df_clean)

    tool_result = None
    if tool == "NewsAPI":
        tool_result = newsapi_top_headlines(os.getenv("NEWSAPI_KEY"), tool_input or query)
    elif tool == "OpenWeather":
        tool_result = openweather_current(os.getenv("OPENWEATHER_KEY"), tool_input or "Madrid")

    return rag_answer, str(tool_result)


with gr.Blocks() as demo:
    gr.Markdown("# RAG + Tools demo")
    with gr.Row():
        query_in = gr.Textbox(label="Consulta")
        tool_sel = gr.Dropdown(choices=["None", "NewsAPI", "OpenWeather"], value="None", label="Tool")
        tool_input = gr.Textbox(label="Tool input (ej: ciudad o tema)")
    out_rag = gr.Textbox(label="RAG respuesta")
    out_tool = gr.Textbox(label="Tool output")
    btn = gr.Button("Enviar")
    btn.click(answer, inputs=[query_in, tool_sel, tool_input], outputs=[out_rag, out_tool])


def main():
    demo.launch()


if __name__ == "__main__":
    main()
