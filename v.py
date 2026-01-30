import cv2

def start_camera():
    # 1. 初始化摄像头，'0' 通常是内置摄像头的索引
    cap = cv2.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头。请检查连接或权限设置。")
        return

    print("摄像头已启动。按下 'q' 键退出程序。")

    while True:
        # 2. 逐帧捕获视频
        ret, frame = cap.read()

        # 如果正确读取帧，ret 为 True
        if not ret:
            print("错误：无法接收帧。")
            break

        # 3. 在窗口中显示结果
        cv2.imshow('Camera Real-time Feed', frame)

        # 4. 监听键盘事件：按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 5. 完成后释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera()