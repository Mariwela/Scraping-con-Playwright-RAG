"""Conjunto de herramientas (tools) que hacen llamadas a APIs externas.

Incluye ejemplos sencillos: Wikipedia summary, NewsAPI headlines y OpenWeather current weather.
Estas funciones devuelven dict con 'success' y 'result' o 'error'.
"""
import os
import requests


def newsapi_top_headlines(api_key: str, query: str = None):
    """Ejemplo que llama a NewsAPI. Requiere una API key.
    Si no hay key, devuelve error amigable.
    """
    if not api_key:
        return {"success": False, "error": "No API key provided for NewsAPI"}
    try:
        params = {"apiKey": api_key, "pageSize": 5}
        if query:
            params["q"] = query
        r = requests.get("https://newsapi.org/v2/top-headlines", params=params, timeout=6)
        r.raise_for_status()
        data = r.json()
        articles = [f"{a.get('title')} - {a.get('source', {}).get('name')}" for a in data.get("articles", [])]
        return {"success": True, "result": articles}
    except Exception as e:
        return {"success": False, "error": str(e)}


def openweather_current(api_key: str, city: str):
    """Devuelve el clima actual de OpenWeather. Requiere API key y nombre de ciudad."""
    if not api_key:
        return {"success": False, "error": "No API key provided for OpenWeather"}
    try:
        params = {"appid": api_key, "q": city, "units": "metric", "lang": "es"}
        r = requests.get("https://api.openweathermap.org/data/2.5/weather", params=params, timeout=6)
        r.raise_for_status()
        data = r.json()
        desc = data.get("weather", [{}])[0].get("description")
        temp = data.get("main", {}).get("temp")
        return {"success": True, "result": f"{city}: {desc}, {temp}°C"}
    except Exception as e:
        return {"success": False, "error": str(e)}
