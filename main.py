import statistics
import time
import pyautogui
from google.cloud import vision
import os
import io
from PIL import Image, ImageDraw
from google.cloud.vision_v1 import AnnotateImageResponse
import json
from difflib import SequenceMatcher
import pytesseract
from navigation_components import download_atomic_cards
from difflib import SequenceMatcher
import Levenshtein
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import AgglomerativeClustering


credential_path = os.environ["GOOGLE_VISION_CREDENTIALS"]
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

with io.open('assets/IMAGE.path', 'rb') as image_file:
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

with open("assets/AtomicCards.json", encoding="UTF-8") as file:
    data = file.read()
atomiccards = json.loads(data)["data"]
referenceCards = atomiccards.keys()
image = Image.open("assets/image.path")
parsed_entries = []
start = time.time()
for i, entry in enumerate(entries):
    if any(Levenshtein.distance(card, entry["description"]) <= 1 for card in referenceCards) and \
            len(entry["description"]) > 1:
        parsed_entries.append(entry)
    else:
        correctCards = sorted(referenceCards, key=lambda x: Levenshtein.distance(x, entry["description"]))[:5]
        correctDistances = [Levenshtein.distance(card, entry["description"]) for card in correctCards]
        draw_box(image, entry["boundingPoly"]["vertices"], "blue")

count = 0
parsed_points = []
for entry in parsed_entries:
    count += 1
    correctCards = [card for card in referenceCards if Levenshtein.distance(card, entry["description"]) <= 1]
    correctDistances = [Levenshtein.distance(card, entry["description"]) for card in correctCards]
    print(entry["description"], "=>", correctCards, "with distances", correctDistances)
    draw_box(image, entry["boundingPoly"]["vertices"], "red")
    parsed_points.append([entry["boundingPoly"]["vertices"][0]["x"], entry["boundingPoly"]["vertices"][0]["y"]])

X = np.array(parsed_points)
# cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
# cluster.fit_predict(X)
# for label, point in zip(cluster.labels_, parsed_points):
#     if label == 0:
#         draw_point(image, {"x": point[0], "y": point[1]}, "red", 15)
#     else:
#         draw_point(image, {"x": point[0], "y": point[1]}, "green", 15)
k_means = KMeans(n_clusters=2, n_init=25,random_state=0).fit(X)
for label, point in zip(k_means.labels_, parsed_points):
    if label == 0:
        draw_point(image, {"x": point[0], "y": point[1]}, "red", 15)
    else:
        draw_point(image, {"x": point[0], "y": point[1]}, "green", 15)
# for center in k_means.cluster_centers_:
#     draw_point(image, {"x": center[0], "y": center[1]}, "green", 30)
print(count)
print(time.time() - start)
fileout = "assets/PROCESSED_IMAGE.path"
image.save(fileout)
