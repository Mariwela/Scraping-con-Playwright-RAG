from scraper import scrape_medal_table
from vector_db import create_vector_db, query_vector_db
from rag import run_rag

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
    run_rag(pregunta, collection, df_clean)

if __name__ == "__main__":
    main()
