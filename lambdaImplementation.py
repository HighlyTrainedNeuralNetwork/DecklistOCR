from dotenv import load_dotenv
import time
from google.cloud import vision
import os
from google.cloud.vision_v1 import AnnotateImageResponse
import json
import Levenshtein
import numpy as np
import base64
from dbscan import MyDBSCAN


class decklistOCR:
    def __init__(self, image):
        self.image = image
        self.referenceCards = []
        self.entries = []
        self.entryCountDict = {}
        self.maindeckEntryCountDict = {}
        self.sideboardEntryCountDict = {}
        self.coordinates = []
        self.cardinality = ""
        self.sideboard_cluster_center = 0

    def getReferenceCards(self):
        with open("assets/AtomicCards.txt", encoding="utf-8") as file:
            for line in file:
                self.referenceCards.append(line.strip())

    def getEntries(self):
        image = vision.Image(content=self.image)
        client = vision.ImageAnnotatorClient()
        response = client.document_text_detection(image=image)
        serializedResponse = AnnotateImageResponse.to_json(response)
        serializedResponse = json.loads(serializedResponse)
        self.entries = sorted(serializedResponse["textAnnotations"][1:], key=lambda x: x["boundingPoly"]["vertices"][0]["x"])

    def joinEntries(self):
        def checkJoin(joinee, row):
            for index, join in enumerate(row):
                if abs(joinee["boundingPoly"]["vertices"][1]["x"] - join["boundingPoly"]["vertices"][0]["x"]) <= 10 \
                        and abs(
                    joinee["boundingPoly"]["vertices"][1]["y"] - join["boundingPoly"]["vertices"][0]["y"]) <= 2 \
                        and join != joinee:
                    join["description"] = joinee["description"] + " " + join["description"]
                    join["boundingPoly"]["vertices"][0] = joinee["boundingPoly"]["vertices"][0]
                    join["boundingPoly"]["vertices"][3] = joinee["boundingPoly"]["vertices"][3]
                    row.pop(row.index(joinee))
                    checkJoin(join, row)
                    return True
                elif 0 < abs(
                        joinee["boundingPoly"]["vertices"][0]["x"] - join["boundingPoly"]["vertices"][1]["x"]) <= 10 \
                        and abs(
                    joinee["boundingPoly"]["vertices"][1]["y"] - join["boundingPoly"]["vertices"][0]["y"]) <= 2 \
                        and join != joinee:
                    join["description"] = join["description"] + " " + joinee["description"]
                    join["boundingPoly"]["vertices"][1] = joinee["boundingPoly"]["vertices"][1]
                    join["boundingPoly"]["vertices"][2] = joinee["boundingPoly"]["vertices"][2]
                    row.pop(row.index(joinee))
                    checkJoin(join, row)
                    return True
        groups = {}
        for entry in self.entries:
            if any(i in groups for i in
                   range(entry["boundingPoly"]["vertices"][0]["y"] - 2, entry["boundingPoly"]["vertices"][0]["y"] + 3)):
                closest = sorted(groups.keys(), key=lambda x: abs(x - entry["boundingPoly"]["vertices"][0]["y"]))[0]
                groups[closest].append(entry)
            else:
                groups[entry["boundingPoly"]["vertices"][0]["y"]] = [entry]
        for group in groups:
            for index, entry in enumerate(groups[group]):
                checkJoin(entry, groups[group])
        self.entries = [entry for group in groups for entry in groups[group]]

    def sematicDistanceBinary(self, candidate, match):
        if ".." in candidate:
            threshold = 0.70
        else:
            threshold = 0.90
        if Levenshtein.ratio(candidate, match) >= threshold:
            return True
        else:
            return False

    def semanticDistanceValue(self, candidate, match):
        return Levenshtein.ratio(candidate, match)

    def processEntries(self):
        if any("Sideboard" in entry["description"] for entry in self.entries):
            match = [entry for entry in self.entries if "Sideboard" in entry["description"]][0]
            self.sideboard_cluster_center = [match["boundingPoly"]["vertices"][0]["x"], match["boundingPoly"]
            ["vertices"][0]["y"]]
        for entry in self.entries:
            entry["description"] = entry["description"].replace("(", "").replace(")", "").replace("[", "") \
                .replace("]", "").replace(",", "")
            entry["description"] = ''.join([char for char in entry["description"] if not char.isdigit()]).strip()
            if any(self.sematicDistanceBinary(entry["description"], card) for card in self.referenceCards) and \
                    len(entry["description"]) > 1:
                entry["description"] = \
                [card for card in self.referenceCards if self.sematicDistanceBinary(entry["description"], card)][0]
                if entry["description"] not in self.entryCountDict.keys():
                    self.entryCountDict[entry["description"]] = [
                        [entry["boundingPoly"]["vertices"][0]["x"], entry["boundingPoly"]["vertices"][0]["y"]]]
                else:
                    self.entryCountDict[entry["description"]].append(
                        [entry["boundingPoly"]["vertices"][0]["x"], entry["boundingPoly"]["vertices"][0]["y"]])
                    ys = [y for x, y in self.entryCountDict[entry["description"]]]
                    if max(ys) - min(ys) > 120:
                        self.entryCountDict[entry["description"]].pop(ys.index(max(ys)))

    def clustering(self):
        self.coordinates = [coord for card in self.entryCountDict for coord in self.entryCountDict[card]]
        x = np.array(self.coordinates)
        clustering = MyDBSCAN(x, eps=236, MinPts=10)
        i = 0
        for card in self.entryCountDict.keys():
            for single in self.entryCountDict[card]:
                if clustering[i] == 1:
                    if card not in self.maindeckEntryCountDict.keys():
                        self.maindeckEntryCountDict[card] = 1
                    else:
                        self.maindeckEntryCountDict[card] += 1
                else:
                    if card not in self.sideboardEntryCountDict.keys():
                        self.sideboardEntryCountDict[card] = 1
                    else:
                        self.sideboardEntryCountDict[card] += 1
                i += 1

    def export(self):
        output = ""
        for card in self.maindeckEntryCountDict.keys():
            output += str(self.maindeckEntryCountDict[card]) + " " + card + "\n"
        output += "\n"
        for card in self.sideboardEntryCountDict.keys():
            output += str(self.sideboardEntryCountDict[card]) + " " + card + "\n"
        return output

def lambda_handler(event, context):
    try:
        start = time.time()
        load_dotenv('.env')
        credential_path = os.getenv('credential_path')
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
        image = base64.b64decode(event["body"])
        decklist = decklistOCR(image)
        decklist.getReferenceCards()
        decklist.getEntries()
        decklist.joinEntries()
        decklist.processEntries()
        decklist.clustering()
        output = decklist.export()
        count = sum(len(decklist.entryCountDict[key]) for key in decklist.entryCountDict)
        output = str(time.time() - start) + "\n" + str(count) + "\n" + output
        return create_response(200, output)
    except Exception as e:
        return create_response(500, "Error: " + str(e))

def create_response(status_code, body=None):
    response = {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
    }
    if body:
        response["body"] = json.dumps(body)
    return response

object = {
  'body': "",
  'isBase64Encoded': True
}
with open('assets/Grixis Phoenix Decklist.png', 'rb') as image_file:
    base64_string = base64.b64encode(image_file.read())
    object["body"] = base64_string
