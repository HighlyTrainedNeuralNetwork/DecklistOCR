import requests
import pytesseract
import pyautogui
from difflib import SequenceMatcher
import json
from bs4 import BeautifulSoup
import requests

def download_atomic_cards():
    atomic_cards = requests.get("https://mtgjson.com/api/v5/AtomicCards.json").text
    with open("assets/AtomicCards.json", "w", encoding="UTF-8") as file:
        file.write(atomic_cards)

