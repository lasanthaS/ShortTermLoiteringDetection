import cv2
import numpy as np

from detectors.DefaultDetector.Constants import Constants, Colors
from detectors.DefaultDetector.models.Person import Person


class Detector:
    # Array of people
    _people = []

    # Person ID
    _personId = 1

    @staticmethod
    def getName():
        return "default"

    def __init__(self, enablePeopleTracking=False):
        self._backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self._enablePeopleTracking = enablePeopleTracking

    def process(self, path):
        frameColored = cv2.imread(path)
        # Converting image into gray-scale mode.
        frame = cv2.cvtColor(frameColored, cv2.COLOR_BGR2GRAY)
        mask = self._calculateForegroundMask(frame)
        contours = self._findContours(mask)

        for contour in contours:
            # Calculating the contour area. Only if the area is greater than a threshold, identified as a person.
            area = cv2.contourArea(contour)
            if area >= Constants.HUMAN_AREA_THRESHOLD:
                cv2.drawContours(frame, contour, -1, Colors.GREEN, 0, 8)

                # Draw bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), Colors.GREEN, 1)

                # Draw the center-point
                cx, cy = Detector._calculateCenter(contour)
                cv2.circle(frame, (cx, cy), 5, Colors.RED, -1)

                # Identify people
                if self._enablePeopleTracking:
                    new = True
                    # Check if the current person already exists.
                    for person in self._people:
                        if abs(x - person.getX()) <= w and abs(y - person.getY()) <= h:
                            new = False
                            person.updateCoords(cx, cy)
                            person.setDim(x, y, w, h)
                            break
                    if new:
                        p = Person(self._personId, cx, cy, 5)
                        p.setDim(x, y, w, h)
                        self._people.append(p)
                        self._personId += 1
            else:
                cv2.drawContours(frame, contour, -1, Colors.BLUE, 2, 8)

        if self._enablePeopleTracking:
            for person in self._people:
                self._trackPerson(person, frame)

        cv2.imshow('Frame', frame)
        cv2.imshow('Foreground Mask', mask)

    @staticmethod
    def _calculateCenter(contour):
        moments = cv2.moments(contour)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return cx, cy

    def _trackPerson(self, person, frame):
        if len(person.getTracks()) >= 2:
            pts = np.array(person.getTracks(), np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, person.getRGB())

        cv2.putText(frame, str(person.getId()), (person.getX(), person.getY()), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    person.getRGB(), 1, cv2.LINE_AA)

    def _calculateForegroundMask(self, frame):
        mask = self._backgroundSubtractor.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # mask = cv2.morphologyEx(binary_image, cv2.MORPH_ERODE, np.ones((5,5), np.uint8))
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        return mask

    def _findContours(self, mask):
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours
