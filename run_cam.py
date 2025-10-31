from ultralytics import YOLO
import cv2
import pandas as pd

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
        results = model(frame, device='cpu', conf=0.3, imgsz=640, half=True, verbose=False)

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

    # Xuất detections ra Excel (chỉ các cột x,y cùng một vài metadata)
    try:
        if len(detections) == 0:
            print("Không có detections để lưu.")
            return
        df = pd.DataFrame(detections)
        out_path = r"D:\Download\Video\detections.xlsx"
        df.to_excel(out_path, index=False)
        print(f"✅ Lưu detections xong: {out_path}")
    except Exception as e:
        print(f"❌ Lỗi khi lưu Excel: {e}")

if __name__ == "__main__":
    main()
