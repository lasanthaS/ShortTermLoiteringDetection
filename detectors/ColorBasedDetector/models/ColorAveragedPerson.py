class ColorAveragedPerson:
    def __init__(self, id=0, averages=(0, 0, 0), lastSeen=-1):
        self.id = id
        self.averages = averages
        self.lastSeen = lastSeen

    def setId(self, id):
        self.id = id

    def getId(self):
        return self.id

    def setAverages(self, averages):
        self.averages = averages

    def getAverages(self):
        return self.averages

    def setLastSeen(self, lastSeen):
        self.lastSeen = lastSeen

    def getLastSeen(self):
        return self.lastSeen

    def __repr__(self):
        return self.averages.__str__()