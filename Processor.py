import sys
import cv2

from utils.Constants import ProcessorConstants as Constants


class Processor:

    def __init__(self, detector, progress=False):
        print("Initializing the processor with {0} detector".format(detector.getName()))

        self.progress = progress
        self.process(detector)

    def process(self, detector):
        print("Start processing...")

        # Iterate all the files in the directory and pass them to the detector.
        for i in range(0, Constants.NUMBER_OF_FRAMES):
            # If progress notifications are enabled, show the progress.
            if self.progress:
                Processor.showProgress("Processing image: {0}".format(i + 1))

            # Build file path and pass it to the detector.
            path = Constants.FILE_PATH_TEMPLATE.replace('{NUMBER}', '%06d' % i)
            detector.process(path, i)

            cv2.waitKey(33)

        cv2.destroyAllWindows()
        print("Completed processing!")

    @staticmethod
    def showProgress(message):
        sys.stdout.write("\r" + message)
        sys.stdout.flush()
