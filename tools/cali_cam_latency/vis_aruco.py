"""
延迟标定：在屏幕上显示 ArUco 序列并记录时间戳。

当 OpenCV 无 GUI 支持时，通过浏览器显示 ArUco。
用法：运行后打开打印的 URL，按 Enter 开始序列。
"""
import cv2
import numpy as np
import time
from datetime import datetime
import csv
import threading
from pathlib import Path

# 全局：当前帧（JPEG bytes），供 Web 服务使用
_current_frame_bytes = None
_frame_lock = threading.Lock()


def _make_frame(aruco_dict, marker_id: int) -> bytes:
    """生成 ArUco 图像，返回 JPEG bytes。"""
    img = np.ones((800, 800), dtype=np.uint8) * 255
    if marker_id is None:
        cv2.putText(img, "Press Enter in terminal to start", (120, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(img, "(then quickly start camera recording)", (100, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    else:
        aruco_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 500)
        cx = (img.shape[1] - aruco_img.shape[1]) // 2
        cy = (img.shape[0] - aruco_img.shape[0]) // 2
        img[cy:cy + aruco_img.shape[0], cx:cx + aruco_img.shape[1]] = aruco_img
        cv2.putText(img, f"ID {marker_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    _, jpg = cv2.imencode(".jpg", img)
    return jpg.tobytes()


def _run_http_server(port: int):
    """后台线程：标准库 http.server，提供当前帧（无 Flask 依赖）。"""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ArUco</title></head>
<body style="margin:0;background:#111;">
<div style="display:flex;justify-content:center;align-items:center;min-height:100vh;">
  <img id="aruco" src="/frame" style="max-width:100%;max-height:100vh;" alt="ArUco" />
</div>
<script>
  setInterval(function(){ var i=document.getElementById('aruco'); i.src='/frame?'+Date.now(); }, 33);
</script>
</body></html>"""

    class ArucoHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/frame":
                with _frame_lock:
                    data = _current_frame_bytes
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

    server = HTTPServer(("0.0.0.0", port), ArucoHandler)
    server.serve_forever()


def generate_aruco_markers(frame_rate: int, max_markers: int, port: int = 8765):
    global _current_frame_bytes

    frame_delay = 1 / frame_rate
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

    # 初始化等待画面
    with _frame_lock:
        _current_frame_bytes = _make_frame(aruco_dict, None)

    # 启动 Web 服务（标准库，无 Flask 依赖）
    t = threading.Thread(target=_run_http_server, args=(port,), daemon=True)
    t.start()
    time.sleep(0.5)

    print()
    print("=" * 60)
    print("  在浏览器打开:  http://127.0.0.1:{}/".format(port))
    print("  若本机访问不了，可试:  http://<本机IP>:{}/".format(port))
    print("=" * 60)
    print()
    print("准备好后，在终端按 Enter 开始 ArUco 序列（ID 0 -> {}）".format(max_markers))
    print("同时启动相机录制，对准此浏览器窗口。")
    print()
    input("按 Enter 开始... ")

    base_dir = Path("data_cali")
    base_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S.%f")[:-3]
    output_dir = base_dir / f"aruco_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "aruco_timestamps.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "timestamp"])
        for counter in range(max_markers + 1):
            unix_time = time.time()
            writer.writerow([counter, unix_time])
            f.flush()

            with _frame_lock:
                _current_frame_bytes = _make_frame(aruco_dict, counter)

            print("ID {} (timestamp {:.6f})".format(counter, unix_time))
            if counter < max_markers:
                time.sleep(frame_delay)

    print()
    print("ArUco 序列结束。数据已保存: {}".format(csv_path))
    print("可关闭浏览器标签页，Ctrl+C 退出程序。")


if __name__ == "__main__":
    generate_aruco_markers(frame_rate=30, max_markers=35)
