import math
import numpy as np
import cv2

from detectors.ColorBasedDetector.Constants import Constants
from detectors.ColorBasedDetector.models.Person import Person


class HumanTracker:
    ADJACENT_MAX_DISTANCE = 100     # Max distance between people between two frames.
    pid = 1                         # Person Id
    frameId = 0                     # Frame Id

    def __init__(self):
        self.people = []  # Initialize people database

    def track(self, frame, cx, cy, x, y, w, h, frameId):
        """
        Track individuals from each contour
        """
        self.frameId = frameId
        new = True
        person = None
        averageColor = self.calculateAverageColor(frame, x, y, w, h)

        # Check if the current person already exists.
        for p in self.people:
            withinDistanceLimit = abs(cx - p.getX()) <= w and abs(cy - p.getY()) <= h
            withinColorRange = self.isInColorRange(p.getColor(), averageColor)

            if withinDistanceLimit and withinColorRange:
                # This person is not a new person. link the person with the old one
                new = False
                p.updateCoords(cx, cy)
                p.setDimension(x, y, w, h)
                p.updateColor(averageColor)
                p.updateLastSeenOn(frameId)
                person = p
                break

        # If this is a new person add it to the people list.
        if new:
            p = Person(self.pid)
            p.updateCoords(cx, cy)
            p.setDimension(x, y, w, h)
            p.updateColor(averageColor)
            p.updateLastSeenOn(frameId)
            self.people.append(p)
            self.pid += 1

        return person

    def drawTracks(self, frame):
        """
        Draw track of each person
        """
        for p in self.people:
            if len(p.getTracks()) < 2:
                continue
            pts = np.array(p.getTracks(), np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, (0, 0, 255))
            if self.frameId - p.getLastSeenOn() < 5:
                cv2.putText(frame, str(p.getId()), (p.getX() + 5, p.getY()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def calculateAverageColor(self, frame, x, y, w, h):
        """
        Calculate color averages.
        """
        n = 3
        hp = math.floor(h / n)
        averages = []
        for i in range(n):
            start = hp * i
            end = h if i == n - 1 else (hp * (i + 1)) - 1
            f = frame[y + start:y + end, x:x + w]
            averages.append(np.average(np.average(f, axis=0), axis=0))
        return averages

    def isInColorRange(self, prev, new):
        """
        Check whether the color difference is within the range.
        """
        t = Constants.COLOR_AVERAGE_THRESHOLD
        valid = True
        for i in range(len(new)):
            valid = valid and new[i] - t <= prev[i] <= new[i] + t
        return valid