import math

import cv2
import numpy as np

from detectors.ColorBasedDetector.Constants import Constants, Colors
from detectors.ColorBasedDetector.models.Person import Person


class Detector:
    # Array of people
    _people = []

    @staticmethod
    def getName():
        return "color-based"

    def __init__(self, enablePeopleTracking=False):
        self._backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self._enablePeopleTracking = enablePeopleTracking

    def process(self, path):
        th = Constants.COLOR_AVERAGE_THRESHOLD

        bgrFrame = cv2.imread(path)

        # Converting BGR image into gray-scale.
        frame = cv2.cvtColor(bgrFrame, cv2.COLOR_BGR2GRAY)
        mask = self._calculateForegroundMask(frame)
        contours = self._findContours(mask)

        # Exposing individuals to calculate averages
        exposedFrame = cv2.subtract(frame, cv2.bitwise_not(mask))

        for contour in contours:

            # Calculating the contour area. Only if the area is greater than a threshold, identified as a person.
            area = cv2.contourArea(contour)
            if area >= Constants.HUMAN_AREA_THRESHOLD:
                # cv2.drawContours(frame, contour, -1, Colors.GREEN, 0, 8)

                # Draw bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), Colors.GREEN, 1)

                # Draw the center-point
                cx, cy = Detector._calculateCenter(contour)
                cv2.circle(bgrFrame, (cx, cy), 5, Colors.RED, -1)

                # Prepare exposed frame for color averaging
                avg = self._calculateAverages(exposedFrame, x, y, w, h)

                # Identify people
                person = None
                if self._enablePeopleTracking:
                    new = True
                    # Check if the current person already exists.
                    for p in self._people:
                        if Detector._isPersonInRange(avg, p.getAverages()):
                            new = False
                            p.updateCoords(cx, cy)
                            p.setDimension(x, y, w, h)
                            p.setAverages(avg)
                            person = p

                    if new:
                        p = Person(cx, cy)
                        p.setDimension(x, y, w, h)
                        p.setAverages(avg)
                        self._people.append(p)
                        person = p

                # Drawing contour and bounding rectangle over the detected person.
                cv2.drawContours(bgrFrame, contour, -1, person.getColor(), 0, 8)
                cv2.rectangle(bgrFrame, (x, y), (x + w, y + h), person.getColor(), 1)
            else:
                # Error: Size is too small to identify as a person
                cv2.drawContours(frame, contour, -1, Colors.BLUE, 2, 8)

        if self._enablePeopleTracking:
            for p in self._people:
                # self.extract_person(person, frame)
                self._trackPerson(p, bgrFrame)

        cv2.imshow('Frame', bgrFrame)
        cv2.imshow('Foreground Mask', mask)

    @staticmethod
    def _calculateCenter(contour):
        moments = cv2.moments(contour)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return cx, cy

    def _calculateAverages(self, frame, x, y, w, h):
        hp = math.floor(h / 3)
        frames = [frame[y:y + hp, x:x + w],
                  frame[y + hp + 1:y + (2 * hp), x:x + w],
                  frame[y + (2 * hp) + 1:y + h, x:x + w]]
        averages = [None, None, None]
        for i in range(len(frames)):
            averages[i] = np.average(np.average(frames[i], axis=0), axis=0)
        return averages

    def _trackPerson(self, person, frame):
        if len(person.getTracks()) >= 2:
            pts = np.array(person.getTracks(), np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, person.getColor())

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

    @staticmethod
    def _isPersonInRange(avg, prevAvg):
        threshold = Constants.COLOR_AVERAGE_THRESHOLD
        isValid = True
        for i in range(len(avg)):
            isValid = isValid and avg[i] - threshold <= prevAvg[i] <= avg[i] + threshold
        return isValid
