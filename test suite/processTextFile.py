def processTextFile(fileName):
    maindeckDict = {}
    sideboardDict = {}
    with open("./testAssets/" + fileName, "r", encoding="UTF-8") as file:
        lines = [line.strip() for line in file.readlines()]
    maindeck = lines[:lines.index("")]
    sideboard = lines[lines.index("") + 1:]
    for line in maindeck:
        count = int(line.split(" ")[0])
        card = " ".join(line.split(" ")[1:])
        maindeckDict[card] = count
    for line in sideboard:
        count = int(line.split(" ")[0])
        card = " ".join(line.split(" ")[1:])
        sideboardDict[card] = count
    return maindeckDict, sideboardDict

def calculateDictDifference(dict1, dict2):
    xorKeys = set(dict1.keys()) ^ set(dict2.keys())
    andKeys = set(dict1.keys()) & set(dict2.keys())
    difference = 0
    for key in xorKeys:
        if key in dict1.keys():
            difference += dict1[key]
        else:
            difference += dict2[key]
    for key in andKeys:
        difference += abs(dict1[key] - dict2[key])
    return difference

print(processTextFile("Phoenix Decklist.txt"))