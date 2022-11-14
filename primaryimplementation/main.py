from decklistOCR import decklistOCR
from dotenv import load_dotenv
import time
import os
import io

load_dotenv('../.env')
image_used = "unknown (3).png"
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
output = decklist.export()
decklist.exportDrawing()
print("Execution took: " + str(time.time() - start) + " seconds.\n" + "Found " + str(count) + " cards.")
print(output)
