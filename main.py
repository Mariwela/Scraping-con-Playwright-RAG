from scraper import scrape_medal_table
from vector_db import create_vector_db, query_vector_db
from rag import run_rag
import os
from agent import answer_with_agent, call_gemini_http

def main():
    print("==========================================")
    print("🏅 PROYECTO SCRAPING OLÍMPICO + RAG")
    print("==========================================\n")

    print("🕸️ Scrapeando datos de Wikipedia...")
    df = scrape_medal_table()
    print(df.head())

    print("\n💾 Creando base de datos vectorial...")
    collection, df_clean = create_vector_db(df)

    print("\n🔍 Consultas de ejemplo:")
    query_vector_db(collection, "¿Qué país ganó más medallas de oro?", df_clean)
    query_vector_db(collection, "¿Qué nación obtuvo más medallas totales?", df_clean)

    pregunta = "¿Qué país ganó más oros?"
    rag_result = run_rag(pregunta, collection, df_clean)

    # If GOOGLE_API_KEY is present, demonstrate the Gemini agent answering using tools + RAG
    if os.getenv("GOOGLE_API_KEY"):
        print("\n🤖 LLM (Gemini) disponible — pidiendo respuesta al agente...")
        final, history = answer_with_agent(pregunta, collection, df_clean)
        print("\n--- Respuesta del agente Gemini ---")
        print(final)
        if history:
            print("\n--- Historial de herramientas usadas ---")
            for h in history:
                print(h)
    else:
        print("\n(No GOOGLE_API_KEY set) — mostrando solo resultado RAG:\n")
        print(rag_result)

if __name__ == "__main__":
    main()
