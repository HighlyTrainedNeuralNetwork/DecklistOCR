import requests
import json

def download_atomic_cards():
    atomic_cards = requests.get("https://mtgjson.com/api/v5/AtomicCards.json").text
    with open("../assets/AtomicCards.json", "w", encoding="UTF-8") as file:
        file.write(atomic_cards)

def trim_atomic_cards():
    with open("../assets/AtomicCards.json", encoding="UTF-8") as file:
        data = file.read()
    atomiccards = json.loads(data)["data"]
    referenceCards = [value[0]["name"] if "faceName" not in value[0].keys() else value[0]["faceName"] for value in
                      atomiccards.values()]
    with open("../assets/AtomicCards.txt", "w", encoding="UTF-8") as file:
        file.write("\n".join(referenceCards))

download_atomic_cards()
trim_atomic_cards()