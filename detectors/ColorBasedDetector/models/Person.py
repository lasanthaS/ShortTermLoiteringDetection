from detectors.ColorBasedDetector.utils.Utils import  Utils


class Person:

    tracks = []

    def __init__(self, cx, cy, maxAge=5):
        self.id = Utils.nextPersonId()
        self.color = Utils.generateColor(self.id)
        self.cx = cx
        self.cy = cy
        self.tracks = []
        self.done = False
        self.state = '0'
        self.age = 0
        self.maxAge = maxAge
        self.dir = None
        self.x = None
        self.y = None
        self.w = None
        self.h = None
        self.avg = [None, None, None]

    def getId(self):
        return self.id

    def getColor(self):
        return self.color

    def getTracks(self):
        return self.tracks

    def getX(self):
        return self.cx

    def getY(self):
        return self.cy

    def getAverages(self):
        return self.avg

    def setAverages(self, avg):
        self.avg = avg

    def updateCoords(self, cx, cy):
        self.age = 0
        self.tracks.append([self.cx, self.cy])
        self.cx = cx
        self.cy = cy

    def setDimension(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

