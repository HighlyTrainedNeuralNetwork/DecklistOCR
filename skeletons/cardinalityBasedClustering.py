import numpy as np
from dbscan import MyDBSCAN

def sideboardCheck(self, point):
    if self.cardinality == "vertical":
        if point[1] <= self.sideboard_cluster_center[1] + 50:
            return True
    else:
        if point[0] >= self.sideboard_cluster_center[0] - 50:
            return True
    return False

def clustering(self):
    if self.sideboard_cluster_center == 0:
        self.coordinates = [coord for card in self.entryCountDict for coord in self.entryCountDict[card]]
        print(self.coordinates)
        x = np.array(self.coordinates)
        print((self.drawOn.size[0] + self.drawOn.size[1]) // 10)
        clustering = MyDBSCAN(x, eps=(self.drawOn.size[0] + self.drawOn.size[1]) // 10, MinPts=10)
        cluster_2 = []
        for label, point in zip(clustering, self.coordinates):
            if label != 0:
                cluster_2.append(point)
        self.sideboard_cluster_center = np.mean(cluster_2, axis=0)
    self.draw_point({"x": self.sideboard_cluster_center[0], "y": self.sideboard_cluster_center[1]}, "magenta", 20)
    points_above = [point for point in self.coordinates if point[1] <= self.sideboard_cluster_center[1]]
    points_right = [point for point in self.coordinates if point[0] >= self.sideboard_cluster_center[0]]
    if len(points_above) < len(points_right):
        self.cardinality = "vertical"
    else:
        self.cardinality = "horizontal"
    print(self.cardinality)
    for card in self.entryCountDict.keys():
        for single in self.entryCountDict[card]:
            isSideboard = self.sideboardCheck(single)
            if isSideboard:
                self.draw_point({'x': single[0], 'y': single[1]}, 'red', 15)
                if card not in self.sideboardEntryCountDict.keys():
                    self.sideboardEntryCountDict[card] = 1
                else:
                    self.sideboardEntryCountDict[card] += 1
            else:
                self.draw_point({'x': single[0], 'y': single[1]}, 'green', 15)
                if card not in self.maindeckEntryCountDict.keys():
                    self.maindeckEntryCountDict[card] = 1
                else:
                    self.maindeckEntryCountDict[card] += 1