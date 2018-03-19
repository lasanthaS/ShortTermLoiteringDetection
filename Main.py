from Processor import Processor
from detectors.DefaultDetector.Detector import Detector as DefaultDetector
from detectors.ColorBasedDetector.Detector import Detector as ColorBasedDetector

# Create an instance of the detector
_defaultDetector = DefaultDetector(enablePeopleTracking=True)
_colorBasedDetector = ColorBasedDetector(enablePeopleTracking=True)

# Start the processor with detector instance
Processor(_colorBasedDetector, verbose=False)
