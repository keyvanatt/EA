import pandas as pd
import time
import random
import logging
import re
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")

# --- CONFIGURATION ---
ACCOUNT_ID = "144440"
OUTPUT_FILE = "resultats_proquest_debug.xlsx"

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Masquage de l'automatisation
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })
    return driver

def search_proquest_debug(driver, marker, year):
    marker_clean = " ".join(re.sub(r'[._]', ' ', marker).split())
    search_url = f"https://www.proquest.com/results.searchresultslist:search?qu={marker_clean}&accountid={ACCOUNT_ID}&fromYear={year}&toYear={year}"
    
    try:
        driver.get(search_url)
        time.sleep(8) # On laisse plus de temps pour le rendu JS
        
        # --- DIAGNOSTIC ---
        # On enregistre ce que voit le robot pour comprendre le blocage
        driver.save_screenshot("debug_screenshot.png")
        logging.info("📸 Capture d'écran effectuée : debug_screenshot.png")

        # Tentative de clic sur les cookies au cas où
        try:
            driver.find_element(By.ID, "onetrust-accept-btn-handler").click()
        except: pass

        # Recherche des titres
        selectors = ["a[id^='resultTitle']", "h3 a", "a.itemTitleLink"]
        for sel in selectors:
            elements = driver.find_elements(By.CSS_SELECTOR, sel)
            if elements:
                return elements[0].get_attribute("href"), elements[0].text
        
        return "N/A", "Structure non trouvée (voir screenshot)"
    except Exception as e:
        return "ERROR", str(e)

def main():
    driver = setup_driver()
    # Test sur la première ligne seulement pour débugger
    try:
        # Remplace par un test manuel ou charge ton DF
        test_marker = "china shineway pharmaceutical group"
        logging.info(f"🔎 Test de recherche : {test_marker}")
        url, title = search_proquest_debug(driver, test_marker, 2025)
        logging.info(f"Résultat : {title} -> {url}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()