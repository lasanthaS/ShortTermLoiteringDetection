class Person:
    def __init__(self, id):
        self.tracks = []
        self.id = id
        self.cx = None
        self.cy = None
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.color = []
        self.lastSeenOn = -1

    def updateCoords(self, cx, cy):
        if self.cx is not None and self.cy is not None:
            self.tracks.append([cx, cy])
        self.cx = cx
        self.cy = cy

    def setDimension(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def getX(self):
        return self.cx

    def getY(self):
        return self.cy

    def getTracks(self):
        return self.tracks

    def getColor(self):
        return self.color

    def updateColor(self, color):
        self.color = color

    def getId(self):
        return self.id

    def updateLastSeenOn(self, frameId):
        self.lastSeenOn = frameId

    def getLastSeenOn(self):
        return self.lastSeenOn