import math

import cv2
import numpy as np

from detectors.ColorBasedDetector.Constants import Constants, Colors


class Detector:
    people = []     # Maintains list of people in the scene all-the-time

    @staticmethod
    def getName():
        """
        Get name of the detector.
        :return:
        """
        return "color-based"

    def __init__(self):
        self.backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    def process(self, path, frameId):
        """
        Process the given frame.
        """
        bgr = cv2.imread(path)  # Read the image from the path
        gs = self.preprocessFrame(bgr) # Pre-process the bgr and generate gray-scale image
        mask = self.fetchForegroundMask(gs) # Fetch foreground mask from the frame

        # Subtract inverted mask from the gray-scale image to get exposed individuals
        exposed = cv2.subtract(gs, cv2.bitwise_not(mask))

        contours = self.findContours(mask)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < Constants.HUMAN_AREA_THRESHOLD:
                continue

            x, y, w, h = cv2.boundingRect(contour)  # Get bounding rectangle
            cx, cy = self.calculatePivot(contour)  # Get pivotal point

            # =======================
            # Identify people
            # =======================
            new = True

            # Check if the current person already exists.
            for p in self.people:
                if abs(x - p.getX()) <= w and abs(y - p.getY()) <= h:
                    new = False
                    # todo set p values
                    p.updateCoords(cx, cy)
                    p.setDimension(x, y, w, h)
                    p.updateColor(self.calculateAverageColor(gs, x, y, w, h))
                    break

            # If this is a new person add it to the people list.
            if new:
                p = Person()
                p.updateCoords(cx, cy)
                p.setDimension(x, y, w, h)
                p.updateColor(self.calculateAverageColor(gs, x, y, w, h))
                self.people.append(p)

            # =======================
            # Draw visual cues
            # =======================

            cv2.drawContours(bgr, contour, -1, (0, 255, 0), 0, 8)       # Contour
            cv2.rectangle(bgr, (x, y), (x + w, y + h), (255, 0, 0), 1)  # Bounding box
            cv2.circle(bgr, (cx, cy), 5, Colors.RED, -1)                # Pivotal point

        for p in self.people:
            self.trackPerson(p, bgr)

        cv2.imshow("bgr", bgr)
        cv2.imshow("mask", mask)

    def preprocessFrame(self, bgr, sharpenImage=False):
        """
        Pre-process the BGR frame.
        """
        # gs = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  # Converting BGR image into gray-scale
        y, cr, cb = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb))   # Converting BGR to YCrCb and split
        gs = cv2.equalizeHist(y)  # Equalizing the histogram
        # gs = cv2.blur(gs, (3, 3))  # Blurring image to reduce noise

        # Sharpen the image
        if sharpenImage:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            gs = cv2.filter2D(gs, -1, kernel)

        return gs

    def fetchForegroundMask(self, gs):
        """
        Fetch foreground mask from the given gray-scale frame.
        """
        mask = self.backgroundSubtractor.apply(gs)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # mask = cv2.morphologyEx(binary_image, cv2.MORPH_ERODE, np.ones((5,5), np.uint8))
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        return mask

    def calculatePivot(self, contour):
        """
        Calcualte the pivot point of a given contour.
        """
        moments = cv2.moments(contour)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return cx, cy

    def findContours(self, mask):
        """
        Find contours within a given mask.
        """
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def trackPerson(self, person, frame):
        """
        Draw track of the person.
        """
        if len(person.getTracks()) < 2:
            return

        pts = np.array(person.getTracks(), np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], False, (0, 0, 255))

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

    def isColorInRange(self, prev, new):
        """
        Check whether the color difference is within the range.
        """
        t = Constants.COLOR_AVERAGE_THRESHOLD
        valid = True
        for i in range(len(new)):
            valid = valid and new[i] - t <= prev[i] <= new[i] + t
        return valid


class Person:
    def __init__(self):
        self.tracks = []
        self.id = None
        self.cx = None
        self.cy = None
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.color = []

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
        return self.x

    def getY(self):
        return self.y

    def getTracks(self):
        return self.tracks

    def getColor(self):
        return self.color

    def updateColor(self, color):
        self.color = color