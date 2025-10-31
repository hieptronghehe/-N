import os
import time
import argparse
from pathlib import Path

import cv2
import pandas as pd


class CoordRecorder:
    def __init__(self, out_path: Path):
        self.out_path = out_path
        self.records = []

    def mouse_callback(self, event, x, y, flags, param):
        # left button click records coordinate
        if event == cv2.EVENT_LBUTTONDOWN:
            # param expected to be a dict with frame_no and timestamp
            info = param if isinstance(param, dict) else {}
            rec = {
                "x": int(x),
                "y": int(y),
                "frame": int(info.get("frame", -1)),
                "time": float(info.get("time", time.time())),
            }
            self.records.append(rec)
            print(f"Recorded: x={rec['x']} y={rec['y']} frame={rec['frame']} time={rec['time']}")

    def save(self):
        if not self.records:
            print("No coordinates recorded. Nothing to save.")
            return
        df = pd.DataFrame(self.records)
        # ensure parent exists
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(self.out_path, index=False)
        print(f"Saved {len(self.records)} coordinates to: {self.out_path}")


def main():
    p = argparse.ArgumentParser(description="Record pixel coordinates by clicking on video frames and save to Excel.")
    p.add_argument("--video", "-v", help="Path to video file. If omitted, opens webcam (0).", default=None)
    p.add_argument("--out", "-o", help="Excel output path", default=r"D:\\Download\\Video\\coords.xlsx")
    args = p.parse_args()

    out_path = Path(args.out)
    recorder = CoordRecorder(out_path)

    # open capture
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: cannot open video source")
        return

    win_name = "frame"
    cv2.namedWindow(win_name)

    frame_no = 0
    paused = False

    print("Instructions:")
    print(" - Left-click on the video to record pixel coordinates.")
    print(" - Press 'p' to pause/resume, 's' to save immediately, 'q' or ESC to quit and save.")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of stream or cannot read frame.")
                break
            frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        display = frame.copy()
        info = {"frame": frame_no, "time": time.time()}

        # set the mouse callback with current frame info
        cv2.setMouseCallback(win_name, recorder.mouse_callback, info)

        cv2.imshow(win_name, display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("p"):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord("s"):
            recorder.save()
        elif key == ord("q") or key == 27:
            # quit and save
            break

    cap.release()
    cv2.destroyAllWindows()
    # final save
    recorder.save()


if __name__ == "__main__":
    main()
from ultralytics import YOLO
import cv2
import csv

def main():
    model_path = r"D:\Download\Video\best1.pt"
    model = YOLO(model_path)

    # Đường dẫn đến file video
    video_path = r"D:\Download\Video\3.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Không mở được video")
        return

    # Danh sách lưu detections (mỗi dòng sẽ chứa frame, label, confidence, x, y)
    detections = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # Khi video kết thúc, quay lại đầu video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Dự đoán
        results = model(frame, device=0, conf=0.3, imgsz=640, half=True, verbose=False)

        for r in results:
            for box in r.boxes:
                # Lấy toạ độ pixel
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                # Tính tâm bbox (x, y)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # Lấy confidence nếu có
                try:
                    conf = float(box.conf[0])
                except Exception:
                    conf = None

                # Lưu thông tin detection (chỉ x,y như yêu cầu)
                detections.append({
                    'frame': frame_idx,
                    'label': label,
                    'confidence': round(conf, 4) if conf is not None else None,
                    'x': cx,
                    'y': cy
                })

                # Vẽ khung + hiển thị thông tin gọn (vẫn giữ để quan sát)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} ({cx},{cy})"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                text_x = cx - text_size[0] // 2
                text_y = y1 - 8 if y1 > 25 else y2 + 25
                cv2.putText(frame, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Live Detection", frame)

        # ESC để thoát
        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # Xuất detections ra CSV (chỉ các cột x,y cùng một vài metadata)
    try:
        if len(detections) == 0:
            print("Không có detections để lưu.")
            return
        out_path = r"D:\Download\Video\detections.csv"
        # Ghi CSV với các cột: frame,label,confidence,x,y
        with open(out_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'label', 'confidence', 'x', 'y'])
            for d in detections:
                writer.writerow([d.get('frame'), d.get('label'), d.get('confidence'), d.get('x'), d.get('y')])
        print(f"✅ Lưu detections xong: {out_path}")
    except Exception as e:
        print(f"❌ Lỗi khi lưu CSV: {e}")

if __name__ == "__main__":
    main()
