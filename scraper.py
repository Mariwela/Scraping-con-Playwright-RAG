from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd

def scrape_medal_table():
    """Hace scraping de la tabla de medallas de Wikipedia (Juegos Olímpicos 2024)"""
    url = "https://en.wikipedia.org/wiki/2024_Summer_Olympics_medal_table"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_load_state("networkidle")
        html = page.content()
        browser.close()

    soup = BeautifulSoup(html, "html.parser")

    # Buscar la tabla de medallas
    table = soup.find("table", {"class": "wikitable"})
    df = pd.read_html(str(table))[0]

    # Normalizar nombres de columnas
    df.columns = ["Rank", "Nation", "Gold", "Silver", "Bronze", "Total"]
    df = df.dropna(subset=["Nation"])
    df = df.reset_index(drop=True)

    return df
