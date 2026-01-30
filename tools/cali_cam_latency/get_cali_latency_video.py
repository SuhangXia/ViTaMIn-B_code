#!/usr/bin/env python3
"""
录制备延迟标定用的相机视频。对准显示 ArUco 的屏幕。

当 OpenCV 无 GUI 时，使用 --headless：浏览器预览 + 终端控制。
"""
import cv2
import numpy as np
import os
import sys
import argparse
import csv
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

# Headless 模式下当前帧
_headless_frame_bytes: Optional[bytes] = None
_headless_lock = threading.Lock()
_headless_port = 8766


def _headless_preview_server(port: int):
    """标准库 http.server，无 Flask 依赖。"""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Camera</title></head>
<body style="margin:0;background:#111;">
<div style="display:flex;justify-content:center;align-items:center;min-height:100vh;">
  <img id="cam" src="/frame" style="max-width:100%;" alt="Camera" />
</div>
<script>setInterval(function(){var i=document.getElementById('cam');i.src='/frame?'+Date.now();},50);</script>
</body></html>"""

    class CamHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/frame":
                with _headless_lock:
                    data = _headless_frame_bytes
                if data:
                    self.send_response(200)
                    self.send_header("Content-type", "image/jpeg")
                    self.send_header("Content-length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_response(204)
                    self.end_headers()
            elif path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML.encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("0.0.0.0", port), CamHandler)
    server.serve_forever()


class VideoRecorder:
    
    def __init__(self, mode: str, camera_device: str, fps: int, width: int, height: int):
        self.mode = mode
        self.camera_device = camera_device
        self.cap = None
        self.video_writer = None
        self.output_dir = None
        self.video_path = None
        self.csv_path = None
        self.frame_timestamps = []
        self.frame_count = 0
        self.fps = fps
        self.width = width
        self.height = height
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
    def setup_output_directory(self) -> bool:
        try:
            base_dir = Path("data_cali")
            base_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S.%f")[:-3]
            folder_name = f"{self.mode}_{timestamp}"
            
            self.output_dir = base_dir / folder_name
            self.output_dir.mkdir(exist_ok=True)
            
            self.video_path = self.output_dir / f"{self.mode}_recording.mp4"
            self.csv_path = self.output_dir / f"{self.mode}_timestamps.csv"
            
            print(f"Output directory created successfully: {self.output_dir}")
            print(f"Video file: {self.video_path}")
            print(f"Timestamp file: {self.csv_path}")
            
            return True
            
        except Exception as e:
            print(f"Failed to create output directory: {e}")
            return False
    
    def initialize_camera(self) -> bool:
        try:
            def _video_sort(p):
                s = str(p).replace("/dev/video", "")
                return int(s) if s.isdigit() else 999
            _video_devs = sorted([str(p) for p in Path("/dev").glob("video*")], key=_video_sort)
            tried = self.camera_device
            self.cap = cv2.VideoCapture(self.camera_device)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = None
                for dev in _video_devs:
                    if dev == self.camera_device:
                        continue
                    cap = cv2.VideoCapture(dev)
                    if cap.isOpened():
                        self.cap = cap
                        self.camera_device = dev
                        print(f"[INFO] {tried} 无法打开，已自动改用 {dev}")
                        break
                    cap.release()
            if not self.cap or not self.cap.isOpened():
                print(f"Cannot open camera {tried}")
                print(f"  当前 /dev 下视频设备: {_video_devs}")
                print(f"  建议: v4l2-ctl --list-devices 查看；关闭可能占用相机的程序后重试")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera initialized successfully:")
            print(f"   Resolution: {width}x{height}")
            print(f"   FPS: {actual_fps:.2f} fps")
            
            self.video_writer = cv2.VideoWriter(
                str(self.video_path), 
                self.fourcc, 
                self.fps, 
                (self.width, self.height)
            )
            
            if not self.video_writer.isOpened():
                print(f"Cannot create video writer")
                return False
            
            return True
            
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            return False
    
    def record_frame(self) -> Tuple[bool, any]:
        ret, frame = self.cap.read()
        
        if ret:
            timestamp = time.time()
            self.frame_timestamps.append({
                'frame_id': self.frame_count,
                'timestamp': timestamp
            })
            
            self.video_writer.write(frame)
            self.frame_count += 1
            
            return True, frame
        
        return False, None
    
    def save_timestamps_csv(self):
        try:
            with open(self.csv_path, 'w', newline='') as csvfile:
                fieldnames = ['frame_id', 'timestamp']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for frame_data in self.frame_timestamps:
                    writer.writerow(frame_data)
            
            print(f"Timestamps saved successfully: {self.csv_path}")
            print(f"Total frames: {len(self.frame_timestamps)}")
            
        except Exception as e:
            print(f"Failed to save timestamps: {e}")
    
    def _run_gui_recording(self) -> bool:
        """原有 GUI 录制逻辑（需 cv2.imshow）。"""
        recording_active = False
        print(f"\nPress 's' to start recording, 's' again to stop, 'q' to quit")
        print("-" * 50)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                return False
            if recording_active:
                timestamp = time.time()
                self.frame_timestamps.append({"frame_id": self.frame_count, "timestamp": timestamp})
                self.video_writer.write(frame)
                self.frame_count += 1
            self.add_info_overlay(frame, recording_active)
            cv2.imshow(f"{self.mode.capitalize()} Recording", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                if not recording_active:
                    recording_active = True
                    print("Recording started...")
                else:
                    recording_active = False
                    print(f"\nRecording completed! Frames: {self.frame_count}")
                    break
        return True

    def _run_headless_recording(self, port: int) -> bool:
        """Headless 模式：浏览器预览 + 终端控制。"""
        global _headless_frame_bytes
        stop_event = threading.Event()

        def wait_for_stop():
            input()  # 阻塞直到用户按 Enter
            stop_event.set()

        th = threading.Thread(target=_headless_preview_server, args=(port,), daemon=True)
        th.start()
        time.sleep(0.5)
        print()
        print("=" * 60)
        print(f"  在浏览器打开:  http://127.0.0.1:{port}/")
        print("  对准显示 ArUco 的浏览器窗口")
        print("=" * 60)
        print()
        print("按 Enter 开始录制，再按 Enter 停止")
        input("准备好后按 Enter 开始... ")
        print("Recording started... (按 Enter 停止)")
        stopper = threading.Thread(target=wait_for_stop, daemon=True)
        stopper.start()
        try:
            while not stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                timestamp = time.time()
                self.frame_timestamps.append({"frame_id": self.frame_count, "timestamp": timestamp})
                self.video_writer.write(frame)
                self.frame_count += 1
                self.add_info_overlay(frame, True)
                _, jpg = cv2.imencode(".jpg", frame)
                with _headless_lock:
                    _headless_frame_bytes = jpg.tobytes()
        except Exception as e:
            print(f"Error: {e}")
            return False
        print(f"\nRecording completed! Frames: {self.frame_count}")
        return True

    def start_recording(self, headless: bool = False, headless_port: int = 8766):
        if not self.setup_output_directory():
            return False
        if not self.initialize_camera():
            return False
        try:
            if headless:
                ok = self._run_headless_recording(headless_port)
                self.cleanup(use_gui=False)
                return ok
            ok = self._run_gui_recording()
            self.cleanup(use_gui=True)
            return ok
        except Exception as e:
            print(f"Error occurred during recording: {e}")
            self.cleanup(use_gui=not headless)
            return False
    
    def add_info_overlay(self, frame, recording_active: bool):
        info_text = f"Frame: {self.frame_count} | Mode: {self.mode.upper()}"
        status_text = "RECORDING" if recording_active else "STANDBY"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        cv2.putText(frame, info_text, (10, 30), font, font_scale, (0, 255, 0), thickness)
        
        color = (0, 255, 0) if recording_active else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 60), font, font_scale, color, thickness)
        
        timestamp_text = f"Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
        cv2.putText(frame, timestamp_text, (10, frame.shape[0] - 20), font, 0.5, (255, 255, 255), 1)
    
    def cleanup(self, use_gui: bool = True):
        try:
            if self.frame_timestamps:
                self.save_timestamps_csv()
            if self.video_writer:
                self.video_writer.release()
            if self.cap:
                self.cap.release()
            if use_gui:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            print("Cleanup completed")
        except Exception as e:
            print(f"Error occurred during cleanup: {e}")


def main():
    ap = argparse.ArgumentParser(description="录制备延迟标定用的相机视频（对准跑 vis_aruco 的屏幕）")
    ap.add_argument("--mode", "-m", default="tactile", help="录制模式名，用于输出目录与文件名")
    ap.add_argument("--device", "-d", default="/dev/video2", help="相机设备路径（USB 相机通常 /dev/video2）")
    ap.add_argument("--fps", "-f", type=int, default=30, help="FPS")
    ap.add_argument("--width", type=int, default=1280, help="宽度")
    ap.add_argument("--height", type=int, default=720, help="高度")
    ap.add_argument("--headless", action="store_true", help="无 GUI 模式（浏览器预览），OpenCV headless 时使用")
    ap.add_argument("--headless-port", type=int, default=8766, help="headless 时 Web 预览端口")
    args = ap.parse_args()

    print(f"Recording mode: {args.mode}")
    print(f"Camera device: {args.device}")
    recorder = VideoRecorder(args.mode, args.device, args.fps, args.width, args.height)

    headless = args.headless
    if not headless:
        try:
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow("_test", test_img)
            cv2.destroyAllWindows()
        except cv2.error:
            print("[INFO] OpenCV 无 GUI 支持，自动使用 --headless 模式")
            headless = True

    success = recorder.start_recording(headless=headless, headless_port=args.headless_port)
    
    if success:
        print(f"\n{args.mode} mode video recording completed successfully!")
        print(f"Output directory: {recorder.output_dir}")
    else:
        print(f"\n{args.mode} mode video recording failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
