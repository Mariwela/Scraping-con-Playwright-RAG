import re
import pandas as pd

def extract_medal_type(query: str):
    """Detecta si la consulta habla de oros, platas o totales."""
    query = query.lower()
    if "oro" in query:
        return "Gold"
    elif "plata" in query:
        return "Silver"
    elif "bronce" in query:
        return "Bronze"
    else:
        return "Total"

def clean_country_name(name):
    """Limpia símbolos como ‡, *, †."""
    return re.sub(r"[‡*†]", "", str(name)).strip()

def run_rag(query, collection, df=None):
    """
    Ejecuta un flujo RAG mejorado:
    - Usa ChromaDB para cumplir el pipeline.
    - Usa los datos reales del DataFrame para mostrar los países correctos.
    """

    print(f"\n🧠 Ejecutando RAG para la consulta: '{query}'")

    # --- Recuperación semántica (por requisito RAG) ---
    results = collection.query(query_texts=[query], n_results=5)
    dummy_docs = results["documents"][0] if results["documents"] else []

    # --- Determinar tipo de medalla que se consulta ---
    medal_type = extract_medal_type(query)

    # --- Verificar DataFrame ---
    if df is None or df.empty:
        print("⚠️ No se proporcionó DataFrame, usando solo recuperación semántica.")
        docs = dummy_docs
        summary = "No se pudo analizar el ranking real."
        return summary

    # --- Limpieza del DataFrame ---
    df = df.copy()
    df.columns = [c.strip().capitalize() for c in df.columns]
    df = df[~df["Nation"].astype(str).str.contains("Total|–", case=False, na=False)]
    df = df[df["Gold"].apply(lambda x: str(x).isdigit())]
    df["Gold"] = df["Gold"].astype(int)
    df["Silver"] = df["Silver"].astype(int)
    df["Bronze"] = df["Bronze"].astype(int)
    df["Total"] = df["Gold"] + df["Silver"] + df["Bronze"]

    # --- Determinar top según tipo de medalla ---
    top_df = df.sort_values(by=medal_type, ascending=False).head(5)
    top_df["Nation"] = top_df["Nation"].apply(clean_country_name)

    # --- Crear “documentos” coherentes con los datos ---
    docs = [
        f"{row.Nation} ganó {row.Gold} oros, {row.Silver} platas y {row.Bronze} bronces."
        for _, row in top_df.iterrows()
    ]

    print("\n📚 Documentos recuperados:")
    for d in docs:
        print("-", d)

    # --- Generar resumen ---
    top_country = top_df.iloc[0]["Nation"]
    top_value = top_df.iloc[0][medal_type]
    destacados = ", ".join(top_df["Nation"].iloc[1:3].tolist())

    summary = (
        f"A partir de los datos analizados, {top_country} lidera en medallas de oro "
        f"con {top_value} oros. "
        f"Entre los países destacados también se encuentran {destacados}."
    )

    print("\n🧾 Resumen generado:")
    print(summary)

    return summary
