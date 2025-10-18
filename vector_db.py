import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

def create_vector_db(df):
    """Crea una base de datos vectorial: usa OpenAI si funciona, o local si no."""

    # 🧹 Limpiar la columna 'Rank'
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")  # convierte a número o NaN
    df = df.dropna(subset=["Rank"])  # elimina filas con Rank no numérico
    df["Rank"] = df["Rank"].astype(int)  # convierte definitivamente a entero

    api_key = os.getenv("OPENAI_API_KEY")
    chroma_client = chromadb.Client()

    embedding_fn = None

    if api_key:
        try:
            print("🔗 Probando OpenAI embeddings...")
            embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key)
            # Verificamos si hay cuota
            embedding_fn(input=["test"])
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

    collection = chroma_client.get_or_create_collection(
        name="olympic_medals",
        embedding_function=embedding_fn
    )

    for idx, row in df.iterrows():
        text = f"{row['Nation']} ganó {row['Gold']} oros, {row['Silver']} platas y {row['Bronze']} bronces."
        collection.add(
            documents=[text],
            ids=[str(idx)],
            metadatas=[{"nation": row["Nation"], "rank": int(row["Rank"])}]
        )

    print("✅ Base de datos vectorial creada correctamente.")
    return collection


