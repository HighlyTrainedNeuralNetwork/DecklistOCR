import time
from google.cloud import vision
import os
import io
from PIL import Image, ImageDraw
from google.cloud.vision_v1 import AnnotateImageResponse
import json
import Levenshtein
import numpy as np
from dotenv import load_dotenv
from dbscan import MyDBSCAN

load_dotenv("../.env")


credential_path = os.getenv('credential_path')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

with io.open('../assets/Grixis Phoenix Decklist.png', 'rb') as image_file:
    content = image_file.read()
image = vision.Image(content=content)

client = vision.ImageAnnotatorClient()
response = client.document_text_detection(image=image)
serializedResponse = AnnotateImageResponse.to_json(response)
serializedResponse = json.loads(serializedResponse)

def checkJoin(joinee, row):
    for index, join in enumerate(row):
        if abs(joinee["boundingPoly"]["vertices"][1]["x"] - join["boundingPoly"]["vertices"][0]["x"]) <= 7 \
                and abs(joinee["boundingPoly"]["vertices"][1]["y"] - join["boundingPoly"]["vertices"][0]["y"]) <= 2 \
                and join != joinee:
            join["description"] = joinee["description"] + " " + join["description"]
            join["boundingPoly"]["vertices"][0] = joinee["boundingPoly"]["vertices"][0]
            join["boundingPoly"]["vertices"][3] = joinee["boundingPoly"]["vertices"][3]
            row.pop(row.index(joinee))
            checkJoin(join, row)
            return True
        elif 0 < abs(joinee["boundingPoly"]["vertices"][0]["x"] - join["boundingPoly"]["vertices"][1]["x"]) <= 7 \
                and abs(joinee["boundingPoly"]["vertices"][1]["y"] - join["boundingPoly"]["vertices"][0]["y"]) <= 2\
                and join != joinee:
            join["description"] = join["description"] + " " + joinee["description"]
            join["boundingPoly"]["vertices"][1] = joinee["boundingPoly"]["vertices"][1]
            join["boundingPoly"]["vertices"][2] = joinee["boundingPoly"]["vertices"][2]
            row.pop(row.index(joinee))
            checkJoin(join, row)
            return True

def draw_point(image, vertice, color, radius):
    draw = ImageDraw.Draw(image)
    draw.ellipse((vertice["x"], vertice["y"], vertice["x"] + radius, vertice["y"] + radius), fill=color, outline=color)
    return image

def draw_box(image, vertices, color):
    """Draw a border around the image using the vertices in the vector list."""
    draw = ImageDraw.Draw(image)
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
    return image

entries = sorted(serializedResponse["textAnnotations"][1:], key=lambda x: x["boundingPoly"]["vertices"][0]["x"])

groups = {}
for entry in entries:
    if any(i in groups for i in range(entry["boundingPoly"]["vertices"][0]["y"] - 2, entry["boundingPoly"]["vertices"][0]["y"] + 3)):
        closest = sorted(groups.keys(), key=lambda x: abs(x - entry["boundingPoly"]["vertices"][0]["y"]))[0]
        groups[closest].append(entry)
    else:
        groups[entry["boundingPoly"]["vertices"][0]["y"]] = [entry]

for group in groups:
    for index, entry in enumerate(groups[group]):
        checkJoin(entry, groups[group])
entries = [entry for group in groups for entry in groups[group]]
for entry in entries:
    entry["description"] = entry["description"].replace("(", "").replace(")", "").replace("[", "")\
        .replace("]", "").replace(",", "").replace(".", "")
    entry["description"] = ''.join([char for char in entry["description"] if not char.isdigit()]).strip()

with open("../assets/AtomicCards.json", encoding="UTF-8") as file:
    data = file.read()
atomiccards = json.loads(data)["data"]
referenceCards = atomiccards.keys()
image = Image.open("../assets/Grixis Phoenix Decklist.png")
print("image size:", image.size)
parsed_entries = []
start = time.time()
for i, entry in enumerate(entries):
    if any(Levenshtein.distance(card, entry["description"]) <= 1 for card in referenceCards) and \
            len(entry["description"]) > 1:
        parsed_entries.append(entry)
    else:
        correctCards = sorted(referenceCards, key=lambda x: Levenshtein.distance(x, entry["description"]))[:5]
        correctDistances = [Levenshtein.distance(card, entry["description"]) for card in correctCards]

count = 0
parsed_points = []
for entry in parsed_entries:
    count += 1
    correctCards = [card for card in referenceCards if Levenshtein.distance(card, entry["description"]) <= 1]
    correctDistances = [Levenshtein.distance(card, entry["description"]) for card in correctCards]
    draw_box(image, entry["boundingPoly"]["vertices"], "red")
    parsed_points.append([entry["boundingPoly"]["vertices"][0]["x"], entry["boundingPoly"]["vertices"][0]["y"]])

X = np.array(parsed_points)
clustering = MyDBSCAN(X, eps=(image.size[0] + image.size[1]) // 10, MinPts=10)
print(clustering)
for label, point in zip(clustering, parsed_points):
    if label == 1:
        draw_point(image, {"x": point[0], "y": point[1]}, "green", 15)
    else:
        draw_point(image, {"x": point[0], "y": point[1]}, "red", 15)
print(count)
print(time.time() - start)
fileout = "../assets/liveModded.png"
image.save(fileout)
