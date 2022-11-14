from primaryimplementation.decklistOCR import decklistOCR
from dotenv import load_dotenv
import time
import os
import io

load_dotenv('../.env')
image_used = "Phoenix Decklist.png"
credential_path = os.getenv('credential_path')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
start = time.time()

with io.open('../assets/' + image_used, 'rb') as image_file:
    content = image_file.read()

decklist = decklistOCR(content)
decklist.getReferenceCards()
decklist.getEntries()
decklist.joinEntries()
decklist.processEntries()
count = sum(len(decklist.entryCountDict[key]) for key in decklist.entryCountDict)
decklist.clustering()
print(decklist.maindeckEntryCountDict)
print(decklist.sideboardEntryCountDict)