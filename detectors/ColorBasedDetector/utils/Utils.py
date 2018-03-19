class Utils:
    currentPersonId = 0

    @staticmethod
    def generateColor(id):
        stepSize = 256 / 8
        colors = [0, 0, 0]
        for i in range(0, id):
            idx = i % 3
            colors[idx] += stepSize
            if colors[idx] > 256:
                colors[idx] = 0
        return colors[0], colors[1], colors[2]

    @staticmethod
    def nextPersonId():
        Utils.currentPersonId += 1
        return Utils.currentPersonId
