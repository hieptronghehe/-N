import os
import time
from datetime import datetime, timezone
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


from ultralytics import YOLO
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

    out_path = r"D:\Download\Video\detections.csv"
    # Initialize CSV: overwrite and write header
    try:
        with open(out_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'label', 'confidence', 'x', 'y', 'time'])
    except Exception as e:
        print(f"❌ Không thể tạo file CSV: {e}")
        return

    frame_idx = 0
    buffer = []  # buffer detections within the current 1s window
    last_save_time = time.time()

    print("Bắt đầu xử lý video, detections sẽ được lưu vào CSV mỗi 1 giây nếu có.")

    while True:
        ret, frame = cap.read()
        if not ret:
            # Nếu muốn chỉ chạy 1 lần đến hết video, break
            print("Kết thúc video.")
            break

        # Dự đoán
        results = model(frame, device=0, conf=0.3, imgsz=640, half=True, verbose=False)

        now = time.time()
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names.get(cls_id, str(cls_id))
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                try:
                    conf = float(box.conf[0])
                except Exception:
                    conf = None

                # Skip unwanted moc labels
                #hgidofdob
                if isinstance(label, str) and label.lower() in {"moc1", "moc2", "moc3", "moc4"}:
                    continue

                buffer.append({
                    'frame': frame_idx,
                    'label': label,
                    'confidence': round(conf, 4) if conf is not None else None,
                    'x': cx,
                    'y': cy,
                    # use global UTC ISO timestamp
                    'time': datetime.now(timezone.utc).isoformat(),
                })

                # Vẽ khung + hiển thị thông tin gọn
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} ({cx},{cy})"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                text_x = cx - text_size[0] // 2
                text_y = y1 - 8 if y1 > 25 else y2 + 25
                cv2.putText(frame, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Live Detection", frame)

        # mỗi 1 giây, ghi buffer vào file CSV (append) và reset buffer
        if time.time() - last_save_time >= 1.0:
            if buffer:
                try:
                    with open(out_path, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        for d in buffer:
                            writer.writerow([d['frame'], d['label'], d['confidence'], d['x'], d['y'], d['time']])
                    print(f"Saved {len(buffer)} detections to {out_path} (timestamp: {last_save_time})")
                except Exception as e:
                    print(f"❌ Lỗi khi ghi CSV: {e}")
                buffer.clear()
            last_save_time = time.time()

        # ESC để thoát
        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_idx += 1

    # write remaining buffer on exit
    if buffer:
        try:
            with open(out_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for d in buffer:
                    writer.writerow([d['frame'], d['label'], d['confidence'], d['x'], d['y'], d['time']])
            print(f"Saved remaining {len(buffer)} detections to {out_path}")
        except Exception as e:
            print(f"❌ Lỗi khi ghi CSV cuối: {e}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
