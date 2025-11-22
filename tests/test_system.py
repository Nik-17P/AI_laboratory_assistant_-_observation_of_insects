import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detectors import VisionDetector
from core.memory import Memory

def test_camera():
    print("Testing camera...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("✅ Camera works")
            cv2.imwrite("test_photo.jpg", frame)
            print("✅ Test photo saved: test_photo.jpg")
        else:
            print("❌ Camera can't read frames")
        cap.release()
    else:
        print("❌ Camera not accessible")

def test_detector():
    print("Testing object detector...")
    detector = VisionDetector()
    test_image = cv2.imread("test_photo.jpg")
    if test_image is not None:
        detections = detector.detect_and_track(test_image)
        print(f"✅ Detector found {len(detections)} objects")
        for det in detections:
            print(f"  - Class: {det['class']}, Confidence: {det['conf']:.2f}")
        
        # Test visualization
        visualized = detector.draw_detections(test_image, detections)
        cv2.imwrite("test_detection.jpg", cv2.cvtColor(visualized, cv2.COLOR_RGB2BGR))
        print("✅ Detection visualization saved: test_detection.jpg")
    else:
        print("❌ Need test photo first")

def test_memory():
    print("Testing memory system...")
    mem = Memory()
    test_id = mem.add_event(
        type_="test",
        summary="Test event summary",
        text="This is a test event for system verification",
        image_path="test_photo.jpg",
        meta={"test": True}
    )
    print(f"✅ Memory test event added with ID: {test_id}")
    
    recent = mem.recent(5)
    print(f"✅ Recent events: {len(recent)}")
    
    results = mem.search("test event")
    print(f"✅ Search found: {len(results)} results")

def test_event_detection():
    print("Testing event detection...")
    from core.detectors import EventDetector
    
    detector = EventDetector()
    test_detections = [
        {'box': [100, 100, 200, 200], 'conf': 0.9, 'class': 'bird', 'is_insect': True},
        {'box': [300, 300, 400, 400], 'conf': 0.8, 'class': 'cat', 'is_insect': True}
    ]
    
    events = detector.analyze(test_detections, None)
    print(f"✅ Event detector found {len(events)} events")
    for event in events:
        print(f"  - Type: {event['type']}, Confidence: {event['confidence']}")

if __name__ == "__main__":
    print("🚀 Starting system tests...")
    test_camera()
    test_detector()
    test_memory()
    test_event_detection()
    print("🎉 All system tests completed!")