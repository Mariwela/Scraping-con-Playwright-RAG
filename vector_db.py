import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

def create_vector_db(df):
    """Crea una base de datos vectorial y limpia los datos de medallas olímpicas."""

    # 🧹 Limpieza general del DataFrame
    df = df.copy()

    # Limpiar símbolos especiales en nombres de países
    df["Nation"] = df["Nation"].astype(str).str.replace(r"[‡\*]", "", regex=True).str.strip()

    # Limpiar y convertir Rank a número
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df = df.dropna(subset=["Rank"])
    df["Rank"] = df["Rank"].astype(int)

    # Convertir las columnas de medallas a enteros
    for col in ["Gold", "Silver", "Bronze", "Total"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Inicializar Chroma y embeddings
    api_key = os.getenv("OPENAI_API_KEY")
    chroma_client = chromadb.Client()

    if api_key:
        try:
            print("🔗 Probando OpenAI embeddings...")
            embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key)
            embedding_fn(input=["test"])  # prueba de cuota
            print("✅ OpenAI embeddings disponibles.")
        except Exception as e:
            print(f"⚠️ Error al usar OpenAI ({e})")
            print("💻 Cambiando a modelo local (SentenceTransformer)...")
            embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
    else:
        print("💻 Usando modelo local (SentenceTransformer)...")
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    # Crear colección en Chroma
    collection = chroma_client.get_or_create_collection(
        name="olympic_medals",
        embedding_function=embedding_fn
    )

    # Añadir documentos limpios
    for idx, row in df.iterrows():
        text = f"{row['Nation']} ganó {row['Gold']} oros, {row['Silver']} platas y {row['Bronze']} bronces."
        collection.add(
            documents=[text],
            ids=[str(idx)],
            metadatas=[{"nation": row["Nation"], "rank": int(row["Rank"])}]
        )

    print("✅ Base de datos vectorial creada correctamente.")
    return collection, df


def query_vector_db(collection, query, df, top_n=3):
    """
    Consulta de ejemplo que devuelve los países correctos según el tipo de medalla.
    """
    query_lower = query.lower()
    if "oro" in query_lower:
        medal_col = "Gold"
    elif "plata" in query_lower:
        medal_col = "Silver"
    elif "bronce" in query_lower:
        medal_col = "Bronze"
    else:
        medal_col = "Total"

    # Copia y limpieza del DataFrame
    df_filtered = df.copy()
    df_filtered = df_filtered[~df_filtered["Nation"].astype(str).str.contains("Total|–", na=False)]

    # Asegurar valores numéricos
    df_filtered[medal_col] = pd.to_numeric(df_filtered[medal_col], errors="coerce").fillna(0).astype(int)

    # Ordenar por la medalla indicada
    top_df = df_filtered.sort_values(by=medal_col, ascending=False).head(top_n)

    print(f"\n🔎 Resultados para: '{query}'\n")
    for _, row in top_df.iterrows():
        print(
            f"🏅 {row['Nation']}: {row['Nation']} ganó {row['Gold']} oros, "
            f"{row['Silver']} platas y {row['Bronze']} bronces, Total: {row['Total']}"
        )
