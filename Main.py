from Processor import Processor
from detectors.DefaultDetector.Detector import Detector as DefaultDetector
from detectors.ColorBasedDetector.Detector import Detector as ColorBasedDetector

# Create an instance of the detector
defaultDetector = DefaultDetector()         # Default color detector
colorBasedDetector = ColorBasedDetector()   # Color based detector

# Start the processor with detector instance
Processor(colorBasedDetector)
