import cv2
import numpy as np

from detectors.ColorBasedDetector.Constants import Constants, Colors
from detectors.ColorBasedDetector.HumanTracker import HumanTracker

class Detector:

    @staticmethod
    def getName():
        """
        Get name of the detector.
        """
        return "color-based"

    def __init__(self):
        self.backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=3000, varThreshold=100)
        self.humanTracker = HumanTracker()

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

            # Identify people
            self.humanTracker.track(gs, cx, cy, x, y, w, h, frameId)

            # Draw visual cues
            cv2.drawContours(bgr, contour, -1, (0, 255, 0), 0, 8)       # Contour
            cv2.rectangle(bgr, (x, y), (x + w, y + h), (255, 0, 0), 1)  # Bounding box
            cv2.circle(bgr, (cx, cy), 5, Colors.RED, -1)                # Pivotal point

        self.humanTracker.drawTracks(bgr)

        cv2.imshow("bgr", bgr)
        # cv2.imshow("mask", mask)


    def preprocessFrame(self, bgr, sharpenImage=False):
        """
        Pre-process the BGR frame.
        """
        # gs = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  # Converting BGR image into gray-scale
        y, cr, cb = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb))   # Converting BGR to YCrCb and split

        # Cretae CLAHE (Contrast Limited Adaptive Histogram Equalization) object
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # gs = clahe.apply(y)

        gs = cv2.equalizeHist(y)  # Equalizing the histogram
        # gs = cv2.blur(gs, (3, 3))  # Blurring image to reduce noise

        # Sharpen the image
        if sharpenImage:
            kernel = np.array([[-1, -1, -1], [-1, 7, -1], [-1, -1, -1]])
            gs = cv2.filter2D(gs, -1, kernel)

        return gs

    def fetchForegroundMask(self, gs):
        """
        Fetch foreground mask from the given gray-scale frame.
        """
        mask = self.backgroundSubtractor.apply(gs)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, np.ones((2, 2), np.uint8))
        # mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((7,7), np.uint8))

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

"""
TODO:
=====
    - Integrate color based detection into the tracking logic
    - Handle exists and re-entries
    - Enhance people identification
    - Track projection and identifying zigzag patterns
    - Movement detection
    - Head pose tracking [HARD]
"""