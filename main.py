from scraper import scrape_medal_table
from vector_db import create_vector_db
from process_data import query_vector_db
from rag import run_rag

def main():
    print("==========================================")
    print("🏅 PROYECTO SCRAPING OLÍMPICO + RAG")
    print("==========================================\n")

    print("🕸️ Scrapeando datos de Wikipedia...")
    df = scrape_medal_table()
    print(df.head())

    print("\n💾 Creando base de datos vectorial...")
    collection = create_vector_db(df)

    print("\n🔍 Consultas de ejemplo:")
    query_vector_db(collection, "¿Qué país ganó más medallas de oro?")
    query_vector_db(collection, "¿Qué nación obtuvo más medallas totales?")

    print("\n🧠 Ejecutando RAG:")
    pregunta = "¿Qué país ganó más oros?"
    run_rag(pregunta, collection, df)

if __name__ == "__main__":
    main()


