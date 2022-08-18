from dotenv import load_dotenv
import time
from google.cloud import vision
import os
import io
from PIL import Image, ImageDraw
from google.cloud.vision_v1 import AnnotateImageResponse
import json
import Levenshtein
import numpy as np
from dbscan import MyDBSCAN


class decklistOCR:
    def __init__(self, image):
        self.image = image
        self.drawOn = 0
        self.referenceCards = []
        self.entries = []
        self.entryCountDict = {}
        self.maindeckEntryCountDict = {}
        self.sideboardEntryCountDict = {}
        self.coordinates = []
        self.cardinality = ""
        self.sideboard_cluster_center = 0

    def draw_point(self, vertice, color, radius):
        if self.drawOn == 0:
            self.drawOn = Image.open(io.BytesIO(self.image))
        draw = ImageDraw.Draw(self.drawOn)
        draw.ellipse((vertice["x"], vertice["y"], vertice["x"] + radius, vertice["y"] + radius), fill=color, outline=color)
        return self.drawOn

    def draw_box(self, vertices, color):
        """Draw a border around the image using the vertices in the vector list."""
        if self.drawOn == 0:
            self.drawOn = Image.open(io.BytesIO(self.image))
        draw = ImageDraw.Draw(self.drawOn)
        draw.polygon(
            [
                vertices[0]["x"],
                vertices[0]["y"],
                vertices[1]["x"],
                vertices[1]["y"],
                vertices[2]["x"],
                vertices[2]["y"],
                vertices[3]["x"],
                vertices[3]["y"],
            ],
            None,
            color,
        )
        return self.drawOn

    def exportDrawing(self):
        self.drawOn.save("assets/drawing.png")

    def getReferenceCards(self):
        with open("assets/AtomicCards.json", encoding="UTF-8") as file:
            data = file.read()
        atomiccards = json.loads(data)["data"]
        self.referenceCards = [value[0]["name"] if "faceName" not in value[0].keys() else value[0]["faceName"] for value in
                          atomiccards.values()]

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
        self.drawOn = Image.open(io.BytesIO(self.image))
        # in theory eps might need to be dynamic. it was previously self.drawOn.size[0] + self.drawOn.size[1]) // 10
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

load_dotenv('.env')
image_used = "Phoenix Decklist.png"
credential_path = os.getenv('credential_path')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
start = time.time()

with io.open('assets/' + image_used, 'rb') as image_file:
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
