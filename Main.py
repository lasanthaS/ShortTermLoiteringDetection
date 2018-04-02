from Processor import Processor
from detectors.DefaultDetector.Detector import Detector as DefaultDetector
from detectors.ColorBasedDetector.Detector import Detector as ColorBasedDetector

# Create an instance of the detector
defaultDetector = DefaultDetector()
colorBasedDetector = ColorBasedDetector()

# Start the processor with detector instance
Processor(colorBasedDetector, verbose=False)
