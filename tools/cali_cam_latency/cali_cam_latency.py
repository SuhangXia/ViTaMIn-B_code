import cv2
import csv
import json
import argparse
from pathlib import Path

def calculate_latency(
    video_path: str,
    aruco_csv: str,
    video_csv: str,
    output_path: str,
    target_id: int = 10,
):
    TARGET_ID = target_id
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    param = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, param)

    aruco_timestamps = {}
    with open(aruco_csv, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_id = int(row['frame_id'])
                timestamp = float(row['timestamp'])
                aruco_timestamps[frame_id] = timestamp
            except ValueError as e:
                print(f"ArUco CSV data error: {row} | Error: {str(e)}")
                exit()

    video_timestamps = {}
    with open(video_csv, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_id = int(row['frame_id'])
                timestamp = float(row['timestamp'])
                video_timestamps[frame_id] = timestamp
            except ValueError as e:
                print(f"Video CSV data error: {row} | Error: {str(e)}")
                exit()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    target_frame = None
    aruco_time = None
    video_time = None
    target_frame_img = None

    print(f"Processing video, searching for ID {TARGET_ID}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None and TARGET_ID in ids.flatten():
            target_frame = frame_count
            target_frame_img = frame.copy()
            
            if TARGET_ID in aruco_timestamps:
                aruco_time = aruco_timestamps[TARGET_ID]
            
            if frame_count in video_timestamps:
                video_time = video_timestamps[frame_count]
            
            print(f"\nFound target ID {TARGET_ID}!")
            print(f"Video frame number: {target_frame}")
            print(f"ArUco timestamp: {aruco_time}")
            print(f"Video timestamp: {video_time}")
            break

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    if target_frame is not None and aruco_time is not None and video_time is not None:
        latency = video_time - aruco_time
        
        video_dir = Path(video_path).parent
        
        if target_frame_img is not None:
            frame_image_path = video_dir / f"target_frame_{TARGET_ID}.jpg"
            cv2.imwrite(str(frame_image_path), target_frame_img)
            
            result_json_path = video_dir / "latency_result.json"
            
            result = {
                "target_frame": target_frame,
                "target_id": TARGET_ID,
                "aruco_timestamp": aruco_time,
                "video_timestamp": video_time,
                "latency_seconds": latency,
                "latency_ms": latency * 1000,
                "fps": fps,
                "total_frames_processed": frame_count,
                "target_frame_image": str(frame_image_path),
                "result_json_path": str(result_json_path)
            }
            
            with open(result_json_path, 'w') as f:
                json.dump(result, f, indent=4)
            
            print("\nLatency calculation completed:")
            print(f"Target frame number: {target_frame}")
            print(f"Target ID: {TARGET_ID}")
            print(f"ArUco timestamp: {aruco_time}")
            print(f"Video timestamp: {video_time}")
            print(f"Latency: {latency:.6f} seconds ({latency * 1000:.3f} milliseconds)")
            print(f"Video FPS: {fps}")
            print(f"Target frame image saved to: {frame_image_path}")
            print(f"Result JSON saved to: {result_json_path}")
        else:
            print("Cannot save target frame image")
            
            result_json_path = video_dir / "latency_result.json"
            
            result = {
                "target_frame": target_frame,
                "target_id": TARGET_ID,
                "aruco_timestamp": aruco_time,
                "video_timestamp": video_time,
                "latency_seconds": latency,
                "latency_ms": latency * 1000,
                "fps": fps,
                "total_frames_processed": frame_count
            }
            
            with open(result_json_path, 'w') as f:
                json.dump(result, f, indent=4)
    else:
        print(f"\nID {TARGET_ID} not found or missing timestamp data")
        print(f"Total frames processed: {frame_count}")
        print("提示: 需在录制期间显示 ArUco（先按录制，再按 vis_aruco 的 Enter 开始序列）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="相机延迟标定：ArUco 在屏幕上按时间显示，相机录制屏幕，用两者时间差算延迟",
        epilog="""示例（传目录更方便）:
  python cali_cam_latency.py --visual_dir data_cali/visual_2026.01.30_16.07.28.336 \\
    --aruco_dir data_cali/aruco_2026.01.30_16.11.51.473
  python cali_cam_latency.py --latest   # 自动用 data_cali 下最新的 visual 和 aruco 目录
"""
    )
    ap.add_argument("--latest", action="store_true", help="自动用 data_cali 下最新的 visual 和 aruco 目录")
    ap.add_argument("--visual_dir", help="visual 录制目录，内含 visual_recording.mp4 和 visual_timestamps.csv")
    ap.add_argument("--aruco_dir", help="aruco 目录，内含 aruco_timestamps.csv")
    ap.add_argument("--video", "-v", help="相机录制的视频路径（与 --aruco_csv --video_csv 一起用）")
    ap.add_argument("--aruco_csv", "-a", help="vis_aruco 的 aruco_timestamps.csv 路径")
    ap.add_argument("--video_csv", "-c", help="录制的 timestamps CSV 路径")
    ap.add_argument("--target_id", "-t", type=int, default=10, help="用于算延迟的 ArUco ID，默认 10")
    ap.add_argument("--output", "-o", default=None, help="结果 JSON 路径；默认存在视频同目录")
    ap.add_argument("--data_cali", default="data_cali", help="--latest 时查找的根目录")
    args = ap.parse_args()

    video_path = None
    aruco_csv_path = None
    video_csv_path = None

    if args.latest:
        base = Path(args.data_cali)
        if not base.exists():
            ap.error(f"--latest 时 {base} 不存在")
        visual_dirs = sorted(base.glob("visual_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        aruco_dirs = sorted(base.glob("aruco_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not visual_dirs or not aruco_dirs:
            ap.error(f"--latest 在 {base} 下未找到 visual_* 或 aruco_* 目录")
        vd = visual_dirs[0]
        ad = aruco_dirs[0]
        mode = vd.name.split("_")[0]
        video_path = str(vd / f"{mode}_recording.mp4")
        video_csv_path = str(vd / f"{mode}_timestamps.csv")
        aruco_csv_path = str(ad / "aruco_timestamps.csv")
        print(f"[--latest] visual: {vd.name}")
        print(f"[--latest] aruco:  {ad.name}")
    elif args.visual_dir and args.aruco_dir:
        vd = Path(args.visual_dir)
        ad = Path(args.aruco_dir)
        mode = vd.name.split("_")[0] if "_" in vd.name else "visual"
        video_path = str(vd / f"{mode}_recording.mp4")
        video_csv_path = str(vd / f"{mode}_timestamps.csv")
        aruco_csv_path = str(ad / "aruco_timestamps.csv")
    elif args.video and args.aruco_csv and args.video_csv:
        video_path = args.video
        aruco_csv_path = args.aruco_csv
        video_csv_path = args.video_csv
    else:
        ap.error("请用 --latest、或 --visual_dir + --aruco_dir、或 --video + --aruco_csv + --video_csv")

    out = args.output
    if not out:
        out = str(Path(video_path).parent / "latency_result.json")

    print(f"Video: {video_path}")
    print(f"ArUco CSV: {aruco_csv_path}")
    print(f"Video CSV: {video_csv_path}")
    print(f"Target ID: {args.target_id}")
    calculate_latency(video_path, aruco_csv_path, video_csv_path, out, target_id=args.target_id)
