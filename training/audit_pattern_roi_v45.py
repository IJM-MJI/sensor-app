from pathlib import Path
import cv2
import numpy as np

from train_models import detect_circle, detect_pattern_roi, resize_for_app

ROOT = Path(r"D:/dual_sensor/dual_sensor/recordings/1")
OUT = Path("training/output/v45-pattern-roi/pattern-roi-audit.jpg")
SAMPLES = [
    ("1_90_H2_only_test.mp4", 4.5),
    ("1_90_H2_only_test_2.mp4", 4.5),
    ("1_90_H2_only_test_3.MOV", 4.5),
    ("1_90_H2_only_4.mp4", 4.5),
    ("1_90_H2_only_5.mp4", 4.5),
    ("1_90_H2O_only_2_extract.mp4", 4.5),
    ("1_90_H2O_only.MOV", 4.5),
    ("1_90_H2O_only_extract_3min.mp4", 4.5),
    ("1_90_RH40_2_x2.mp4", 45.0),
    ("1_90_RH40_3.mp4", 90.0),
    ("1_90_RH40_4.mp4", 110.0),
    ("1_90_RH40_5_x2.mp4", 90.0),
]


def frame_at(path: Path, seconds: float):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def annotate(name: str, frame: np.ndarray) -> np.ndarray:
    small = resize_for_app(frame)
    result = detect_pattern_roi(frame)
    if result is None:
        try:
            result = (*detect_circle(frame), "hough-fallback")
        except RuntimeError:
            result = None
    canvas = small.copy()
    if result is not None:
        cx, cy, r, source = result
        cv2.circle(canvas, (cx, cy), r, (40, 40, 255), 2)
        cv2.circle(canvas, (cx, cy), round(.90 * r), (0, 220, 255), 2)
        cv2.rectangle(canvas, (round(cx-.55*r), round(cy-.62*r)),
                      (round(cx+.35*r), round(cy+.14*r)), (0, 145, 255), 2)
        cv2.rectangle(canvas, (round(cx-.55*r), round(cy+.18*r)),
                      (round(cx+.35*r), round(cy+.68*r)), (255, 210, 0), 2)
        status = f"{source} ({cx},{cy}) r={r}"
    else:
        status = "NO PATTERN ROI"
    target = np.zeros((360, 480, 3), np.uint8)
    scale = min(480/canvas.shape[1], 320/canvas.shape[0])
    shown = cv2.resize(canvas, (round(canvas.shape[1]*scale), round(canvas.shape[0]*scale)))
    x0=(480-shown.shape[1])//2;y0=34+(320-shown.shape[0])//2
    target[y0:y0+shown.shape[0],x0:x0+shown.shape[1]]=shown
    cv2.putText(target, name, (8,16), cv2.FONT_HERSHEY_SIMPLEX, .42, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(target, status, (8,33), cv2.FONT_HERSHEY_SIMPLEX, .42, (255,255,255), 1, cv2.LINE_AA)
    return target


tiles=[]
for name, seconds in SAMPLES:
    frame=frame_at(ROOT/name, seconds)
    tiles.append(annotate(name, frame) if frame is not None else np.zeros((360,480,3),np.uint8))
montage=np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)])
OUT.parent.mkdir(parents=True,exist_ok=True)
cv2.imwrite(str(OUT),montage,[cv2.IMWRITE_JPEG_QUALITY,92])
print(OUT)
