import cv2
import numpy as np
import math

from detectors.ColorBasedDetector.models.Person import Person
from detectors.ColorBasedDetector.Constants import Constants, Colors


class Detector:

    # Array of people
    _people = []

    # Person ID
    _personId = 1

    @staticmethod
    def getName():
        return "color-based"

    def __init__(self, enablePeopleTracking=False):
        self._backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self._enablePeopleTracking = enablePeopleTracking

    def process(self, path):
        T = Constants.COLOR_AVERAGE_THRESHOLD

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
                cv2.drawContours(frame, contour, -1, Colors.GREEN, 0, 8)

                # Draw bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), Colors.GREEN, 1)

                # Draw the center-point
                cx, cy = Detector._calculateCenter(contour)
                cv2.circle(frame, (cx, cy), 5, Colors.RED, -1)

                # Prepare exposed frame for color averaging
                a1, a2, a3 = self._calculateAverages(exposedFrame, x, y, w, h)

                # Identify people
                if self._enablePeopleTracking:
                    new = True
                    # Check if the current person already exists.
                    for person in self._people:
                        # person = self._people.get(id)
                        pa1, pa2, pa3 = person.getAverages()
                        if (a1 - T <= pa1 <= a1 + T) and (a2 - T <= pa2 <= a2 + T) and (a3 - T <= pa3 <= a3 + T):
                            new = False
                            person.updateCoords(cx, cy)
                            person.setDim(x, y, w, h)
                            person.setAverages(a1, a2, a3)

                    if new:
                        p = Person(self._personId, cx, cy, 5)
                        p.setDim(x, y, w, h)
                        p.setAverages(a1, a2, a3)
                        self._people.append(p)
                        self._personId += 1
            else:
                cv2.drawContours(frame, contour, -1, Colors.BLUE, 2, 8)

        if self._enablePeopleTracking:
            for person in self._people:
                # self.extract_person(person, frame)
                self._trackPerson(person, frame)

        # print(self._colorAveragedPeople)

        cv2.imshow('Frame', frame)
        cv2.imshow('Foreground Mask', mask)

    @staticmethod
    def _calculateCenter(contour):
        moments = cv2.moments(contour)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return cx, cy

    def _calculateAverages(self, frame, x, y, w, h):
        hp =  math.floor(h / 3)
        head = frame[y:y + hp, x:x + w]
        torso = frame[y + hp + 1:y + (2 * hp), x:x + w]
        limbs = frame[y + (2 * hp) + 1:y + h, x:x + w]

        avgHead = np.average(np.average(head, axis=0), axis=0)
        avgTorso = np.average(np.average(torso, axis=0), axis=0)
        avgLimbs = np.average(np.average(limbs, axis=0), axis=0)

        return avgHead, avgTorso, avgLimbs


    def _trackPerson(self, person, frame):
        if len(person.getTracks()) >= 2:
            pts = np.array(person.getTracks(), np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, person.getRGB())

        # cv2.putText(frame, str(person.getId()), (person.getX(), person.getY()), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
        #             person.getRGB(), 1, cv2.LINE_AA)

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


