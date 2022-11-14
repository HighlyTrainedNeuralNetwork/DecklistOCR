from primaryimplementation.decklistOCR import decklistOCR
from dotenv import load_dotenv
import time
import os
import io
from processTextFile import processTextFile, calculateDictDifference
from os import listdir
from os.path import isfile, join

load_dotenv('../.env')
credential_path = os.getenv('credential_path')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
start = time.time()

images = [f for f in listdir("testAssets") if isfile(join("testAssets", f)) and f.split(".")[1] == "png"]

for image_used in images:
    print("Testing " + image_used.split(".")[0] + "...")
    with io.open('testAssets/' + image_used, 'rb') as image_file:
        content = image_file.read()
    decklist = decklistOCR(content)
    decklist.getReferenceCards()
    decklist.getEntries()
    decklist.joinEntries()
    decklist.processEntries()
    count = sum(len(decklist.entryCountDict[key]) for key in decklist.entryCountDict)
    decklist.clustering()
    validationMaindeck, validationSideboard = processTextFile(image_used.split(".")[0] + ".txt")
    maindeckDifference = calculateDictDifference(decklist.maindeckEntryCountDict, validationMaindeck)
    sideboardDifference = calculateDictDifference(decklist.sideboardEntryCountDict, validationSideboard)
    print("Off by " + str(maindeckDifference) + " cards in maindeck.")
    print("Off by " + str(sideboardDifference) + " cards in sideboard.")