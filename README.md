# Scraping-con-Playwright-RAG
Scraping de una página de Juegos Olýmpicos, almacena esta información en una base de datos vectorial (ChromaDB) y usa RAG para dar respuestas sobre los rankings de medallas
##1. EXTRACCIÓN DE DATOS - scraper.py
https://en.wikipedia.org/wiki/2024_Summer_Olympics_medal_table
Utiliza la biblioteca playwright para abrir una instancia de navegador (headless) y navegar a la página de la tabla de medallas de los Juegos Olímpicos de Verano 2024 en Wikipedia.

Una vez que la página está cargada, obtiene el código HTML.

Utiliza BeautifulSoup para analizar el HTML y encontrar la tabla de medallas específica (clase wikitable).

pandas se utiliza para leer directamente el HTML de la tabla y convertirlo en un DataFrame, una estructura de datos tabular conveniente para el procesamiento.

Normaliza las columnas a ["Rank", "Nation", "Gold", "Silver", "Bronze", "Total"] y devuelve el DataFrame.

##2. ALMACENAMIENTO VECTORIAL - vector_db.py
Limpia el DataFrame de pandas de símbolos especiales y convierte las columnas de medallas y Rank a tipos de datos enteros.

Embeddings: Configura la función de embedding para convertir el texto en vectores:

Intenta usar la función de embedding de OpenAI si la clave API está disponible.

Si falla o la clave no está, usa un modelo de lenguaje local con SentenceTransformerEmbeddingFunction.

Crea una colección llamada "olympic_medals" en ChromaDB.

Itera sobre cada fila del DataFrame y crea un texto descriptivo. Este texto se convierte en un vector (embedding) y se almacena junto con metadatos en la base de datos vectorial ChromaDB.

Hace una consulta de demostración simple.
##3. RAG - rag.py
Obtiene los documentos vectoriales más relevantes a la consulta del usuario.

Analiza la consulta, valida datos y ordena el DataFrame real según el tipo de medalla extraído

Genera nuevos "documentos" de contexto a partir de los países con mejores resultados del DataFrame.

Generación del Resumen: Utiliza los datos del DataFrame ordenado para construir una respuesta final textual que identifica al país líder y a otros destacados.
##4. ORQUESTACIÓN - main.py
Orquesta todas las operaciones: obtiene los datos, crea la base de datos vectorial en ChromaDB, muestra consultas simples basadas en el ranking del DataFrame y demuestra el flujo RAG: consulta semántica + extracción del top real + generación de respuesta.

##5. process_data.py
Muestra los resultados de la consulta semántica en ChromaDB sin la lógica de ordenar el DataFrame por tipo de medalla.
