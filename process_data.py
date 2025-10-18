def query_vector_db(collection, query_text):
    """Consulta semántica en la base de datos vectorial"""
    results = collection.query(
        query_texts=[query_text],
        n_results=3
    )

    print(f"\n🔎 Resultados para: '{query_text}'\n")
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"🏅 {meta['nation']}: {doc}")
