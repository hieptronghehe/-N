import os
import time
from datetime import datetime, timezone
import argparse
from pathlib import Path

import cv2
import pandas as pd
import numpy as np
from chuyendoitoado import get_projection_matrix


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
import db_manage


def main():
    model_path = r"D:\Download\Video\best1.pt"
    model = YOLO(model_path)
    # load projection matrix (homography) to convert pixel -> project coordinates
    proj_matrix = get_projection_matrix().astype('float32')

    # Đường dẫn đến file video
    video_path = r"D:\Download\Video\3.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Không mở được video")
        return

    # Prepare DB
    db_path = r"D:\Download\Video\data.db"
    # ensure parent folder exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    try:
        db_manage.create_temp_table(db_path)
    except Exception as e:
        print(f"❌ Không thể tạo/kiểm tra database: {e}")
        return

    frame_idx = 0
    buffer = []  # buffer detections within the current 1s window
    last_save_time = time.time()

    print("Bắt đầu xử lý video, detections sẽ được lưu vào database mỗi 1 giây nếu có.")

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

                # Skip unwanted moc labels (case-insensitive)
                if label.lower() in {"moc1", "moc2", "moc3", "moc4"}:
                    continue

                # Transform pixel center (cx,cy) -> project coordinates using homography
                try:
                    pt = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
                    tpt = cv2.perspectiveTransform(pt, proj_matrix)
                    tx, ty = float(tpt[0, 0, 0]), float(tpt[0, 0, 1])
                except Exception as e:
                    # If transform fails, fall back to pixel coords
                    print(f"⚠️ Transform failed for point ({cx},{cy}): {e}")
                    tx, ty = float(cx), float(cy)

                buffer.append({
                    'frame': frame_idx,
                    'label': label,
                    'confidence': round(conf, 4) if conf is not None else None,
                    'X': tx,
                    'Y': ty,
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

        # mỗi 1 giây, ghi buffer vào database (append) và reset buffer
        if time.time() - last_save_time >= 1.0:
            if buffer:
                try:
                    # Prepare list of (x_location, y_location, person_ID)
                    db_rows = []
                    person_map = {'songoku': 1, 'dog': 2}
                    for d in buffer:
                        try:
                            x_val = int(round(float(d.get('X', 0))))
                        except Exception:
                            x_val = None
                        try:
                            y_val = int(round(float(d.get('Y', 0))))
                        except Exception:
                            y_val = None
                        if x_val is None or y_val is None:
                            continue
                        lbl = str(d.get('label', '')).lower()
                        if lbl in {"moc1", "moc2", "moc3", "moc4"}:
                            continue
                        person_id = person_map.get(lbl)
                        # person_id may be None -> inserted as NULL
                        db_rows.append((x_val, y_val, person_id))

                    if db_rows:
                        db_manage.add_many_temp(db_path, db_rows)
                        print(f"Inserted {len(db_rows)} rows into DB: {db_path} (timestamp: {last_save_time})")
                    else:
                        print("No valid rows to insert into DB")
                except Exception as e:
                    print(f"❌ Lỗi khi ghi DB: {e}")
                buffer.clear()
            last_save_time = time.time()

        # ESC để thoát
        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_idx += 1

    # write remaining buffer on exit -> insert into DB
    if buffer:
        try:
            db_rows = []
            person_map = {'songoku': 1, 'dog': 2}
            for d in buffer:
                try:
                    x_val = int(round(float(d.get('X', 0))))
                except Exception:
                    x_val = None
                try:
                    y_val = int(round(float(d.get('Y', 0))))
                except Exception:
                    y_val = None
                if x_val is None or y_val is None:
                    continue
                lbl = str(d.get('label', '')).lower()
                if lbl in {"moc1", "moc2", "moc3", "moc4"}:
                    continue
                person_id = person_map.get(lbl)
                db_rows.append((x_val, y_val, person_id))

            if db_rows:
                db_manage.add_many_temp(db_path, db_rows)
                print(f"Inserted remaining {len(db_rows)} rows into DB: {db_path}")
            else:
                print("No remaining valid rows to insert into DB")
        except Exception as e:
            print(f"❌ Lỗi khi ghi DB cuối: {e}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
