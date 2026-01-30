# 标定使用说明：手眼标定 + 延迟标定

## 一、ArUco 码准备（你已有 ID 10 大码）

- **手眼标定**用**打印的** ArUco，ID 10 即可，无需再打印别的。
- 在 `aruco_config_finray.yaml`（或你用的 aruco yaml）里确保有 `10: <尺寸米>`。  
  **尺寸 = 黑色边框一条边的实际长度（米）**。例如 50mm → `0.05`，100mm → `0.1`。量一下你的大码，改 yaml 里 `marker_size_map` 的 `10`。
- 若用 **DICT_4X4_50**，ID 0–49 都支持，10 在内。

---

## 二、手眼标定（T_CameraToQuestHand）

**目的**：算相机坐标系到 Quest 手柄坐标系的**固定**变换矩阵，不依赖机械臂，只靠 ArUco + Quest。

**手眼一体（eye-in-hand）**：若你的**相机和 Quest 左手柄固连**在一起（装在同一个采集装置上），即「手在眼睛上」、二者作为刚体一起动，本算法就是按这种 setup 来的。ArUco **固定在场景**里不动，你拿着「相机+手柄」一体多换姿态；解出的 `T_CameraToQuestHand` 即二者之间的安装关系。**左手柄**请用 `--hand left`。

### 方式 A：按键采集 + 求解（推荐，无延迟）

**一键脚本** `collect_and_solve_hand_eye.py`：按 **`s`** 同时拍一张照片 + 记录当前 Quest 位姿，采集满 N 对后自动 SVD 求解，直接打印 **T_quest2Camera**。按下时尽量保持不动，无同步延迟问题。

```bash
# 1. 先执行 adb forward tcp:7777 tcp:7777
# 2. 在 Quest 上启动 Unity 应用（发送位姿）
# 3. 固定 ArUco，相机+手柄对准，再运行脚本（脚本会等待连接，最多 120s）
python tools/cali_hand_eye/collect_and_solve_hand_eye.py \
  --hand left --aruco_id 10 --num_pairs 10 \
  --output_dir ./hand_eye_calib_data
# 相机默认从 config/task_config.yaml 的 calculate_width.hand_eye_camera 读（已设为 /dev/video2）；可 --camera 覆盖
# 启动时会打印 [Camera] 使用设备 /dev/videoX，请确认是 USB 相机而非笔记本摄像头
```

**Quest 通讯**：与 `00_get_pose` 相同，TCP localhost:7777，JSON 每行一条，含 `left_wrist`/`right_wrist`，各带 `position`{x,y,z}、`rotation`{x,y,z,w}。

**若连接失败，先验证**：运行 `python vitamin_b_data_collection_pipeline/00_get_pose.py --cfg config/task_config.yaml`。若 00_get_pose 能显示 FPS 并收包，则通讯正常；否则检查 adb forward、Quest 上的 Unity 是否在跑。采集脚本默认打印 Quest 连接/收包/解析的调试信息；加 `--no-debug` 可关闭。

- `--num_pairs`：采集对数，默认 10，至少 4
- `--camera`：相机设备，不指定则用 config 的 `calculate_width.hand_eye_camera`（已配为 `/dev/video2` USB 相机）
- `--marker_size_m`：ArUco 边长（米），如 150mm → `0.15`，覆盖 yaml 配置
- 内参：默认从 `config/task_config.yaml` 的 `calculate_width.cam_intrinsic_json_path` 读；也可 `--intrinsics_json` 指定。JSON 需为 fisheye 格式（intrinsic_type: FISHEYE, radial_distortion_1..4 等）
- 照片和位姿保存在 `--output_dir`，求解结果打印并保存为 `T_quest2camera.npy`

### 方式 B：视频 + Quest JSON（需后处理同步）

1. **固定 ArUco**：把 ID 10 的码**固定在场景**里（桌面、墙壁等）。相机+手柄一体移动，相机始终能看到码。
2. **同时录两样**：
   - **视觉相机**视频（能看到 ArUco 的那路）。
   - **Quest 位姿**：运行 `00_get_pose`，保存到 `data/<任务>/all_trajectory/` 的 `quest_poses_*.json`。

```bash
# 例：先建任务目录，再录 Quest 位姿
mkdir -p data/calib/all_trajectory
python vitamin_b_data_collection_pipeline/00_get_pose.py --cfg config/task_config.yaml
# 按说明连接 Quest、adb forward，跑起来后选保存到 data/calib/all_trajectory（或你用的 task 名）
```

3. 同一次录制里，**边录 Quest 边录视觉视频**，多换几个「相机+手柄」一体的姿态（绕 ArUco 转、远近移动等），时长至少几十秒，保证时间重叠。

### 2. 运行手眼标定脚本

**Quest JSON 放在哪，脚本就从哪找**：

- 按** pipeline 习惯**：`data/<任务>/all_trajectory/` 下有 `quest_poses_*.json`。  
  传 `--quest_dir data/<任务>` 或 `--quest_dir data/<任务>/all_trajectory` 都可以，脚本会递归或查 `all_trajectory`。
- 或直接指定文件：`--quest_jsons path/to/quest_poses_xxx_part001.json path/to/part002.json ...`

**示例 1：用视频 + Quest 目录（递归 / all_trajectory）**

```bash
python tools/cali_hand_eye/cali_camera_quest_hand.py \
  --quest_dir ./data/calib \
  --video ./data/calib/left_hand_visual.mp4 \
  --intrinsics_json ./assets/intri_result/gopro_intrinsics_2_7k.json \
  --aruco_yaml ./assets/aruco_config_finray.yaml \
  --aruco_id 10 \
  --hand left \
  --search_time_offset \
  --method opencv \
  --output ./assets/tf_cali_result/T_CameraToQuestHand_left.npy
```

若 Quest 在 `data/my_task/all_trajectory/`：

```bash
--quest_dir ./data/my_task
```

**示例 2：显式指定 Quest JSON**

```bash
python tools/cali_hand_eye/cali_camera_quest_hand.py \
  --quest_jsons ./data/calib/all_trajectory/quest_poses_2025.01.01_12.00.00_part001.json \
  --video ./data/calib/left_hand_visual.mp4 \
  --intrinsics_json ./assets/intri_result/gopro_intrinsics_2_7k.json \
  --aruco_yaml ./assets/aruco_config_finray.yaml \
  --aruco_id 10 \
  --hand left \
  --search_time_offset \
  --output ./assets/tf_cali_result/T_CameraToQuestHand_left.npy
```

**常用参数**：

- `--aruco_id 10`：你现有的大码。
- `--hand left` / `right`：对应左手/右手。
- `--search_time_offset`：自动搜 Quest 与视频的时间偏移，建议加上。
- `--method opencv`（默认）或 `svd`。

**常见报错**：  
- **「目录不存在」**：`--quest_dir` 指向的目录没建。先 `mkdir -p data/calib/all_trajectory`（或你的任务名），再跑 `00_get_pose` 并选保存到该处。  
- **「未找到 quest_poses_*.json」**：1）已跑 `00_get_pose` 并保存；2）保存目录即 `--quest_dir`（或其下 `all_trajectory`）或 `--quest_jsons` 里的路径；3）该目录下确有 `quest_poses_*.json`。

---

## 三、延迟标定（相机 latency，paper 里的做法）

**目的**：测**相机从「看到画面」到「该帧时间戳」的延迟**。做法：屏幕按时间显示 ArUco，相机拍屏幕，用「ArUco 显示时间」与「视频帧时间」差得到延迟。

### 1. 你要配合做的

- **vis_aruco**：在电脑屏幕上按固定节奏显示 ArUco 序列（ID 0, 1, 2, …），并打 `aruco_timestamps.csv`（frame_id = ArUco ID，timestamp = 显示时刻）。
- **get_cali_latency_video**：用**待标定的相机**对准**该屏幕**录像，同时打视频的 `*_timestamps.csv`（frame_id, timestamp）。
- **cali_cam_latency**：读上述两个 CSV + 视频，找到「第一次出现某 ArUco ID」的帧，用 `视频帧时间 - ArUco 显示时间` 算延迟。

**不需要打印 ArUco**；延迟标定用的是**屏幕上的 ArUco**。

### 2. 三步操作（按顺序）

**Step 1：运行 vis_aruco（显示 ArUco + 记时）**

```bash
python tools/cali_cam_latency/vis_aruco.py
```

- 启动后**在浏览器打开**打印的 URL（如 `http://127.0.0.1:8765/`），会显示 ArUco 画面。若 OpenCV 无 GUI（如 opencv-python-headless），本脚本使用 Web 显示。
- 准备好后，在**终端**按 **Enter** 开始序列，会依次显示 ID 0, 1, 2, … 到 35，并写 `data_cali/aruco_<时间>/aruco_timestamps.csv`。
- 用 **ID 10** 做延迟时，默认已包含。
- 保持浏览器窗口可见，进行 Step 2。

**Step 2：运行 get_cali_latency_video（相机录屏）**

```bash
python tools/cali_cam_latency/get_cali_latency_video.py \
  --mode visual --device /dev/video2 --fps 30 --width 1280 --height 720
```

- `--mode`：录制模式名，影响输出目录与文件名；视觉相机用 `visual`。
- `--device`：相机设备（USB 相机通常 `/dev/video2`）。
- 若 OpenCV 无 GUI（opencv-python-headless），脚本会自动用 **headless 模式**：在浏览器打开打印的 URL 预览画面，终端按 Enter 开始/停止录制。
- 相机对准 **显示 ArUco 的浏览器窗口**，按 **`s`**（或 Enter，headless 时）开始/停止录制。
- 输出在 `data_cali/<mode>_<时间>/`：`<mode>_recording.mp4` 和 `<mode>_timestamps.csv`。

**Step 3：运行 cali_cam_latency（算延迟）**

**推荐：传目录**（把下面路径换成你 data_cali 下实际的目录名）：

```bash
python tools/cali_cam_latency/cali_cam_latency.py \
  --visual_dir data_cali/visual_2026.01.30_16.07.28.336 \
  --aruco_dir data_cali/aruco_2026.01.30_16.11.51.473 \
  --target_id 10
```

**或自动用最新**：

```bash
python tools/cali_cam_latency/cali_cam_latency.py --latest
```

- `--visual_dir`：Step 2 的 visual 录制目录（内含 `visual_recording.mp4` 和 `visual_timestamps.csv`）。
- `--aruco_dir`：Step 1 的 aruco 目录（内含 `aruco_timestamps.csv`）。
- `--target_id 10`：用 ID 10 出现的那一帧算延迟。

**⚠️ 正确时序（否则会 "ID 10 not found"）**：  
1. **先**启动 vis_aruco、打开浏览器、再启动 get_cali_latency_video、对准浏览器。  
2. 在**相机录制**终端按 Enter **开始录制**。  
3. **立刻**在 vis_aruco 终端按 Enter **开始 ArUco 序列**（必须在录制期间显示）。  
4. 序列结束后，在录制终端按 Enter 停止。  
- 若先录完再跑 vis_aruco，视频里没有 ArUco，必然无法算出延迟。

结果会写在视频同目录的 `latency_result.json`（或 `--output` 指定路径），里面有 `latency_seconds`、`latency_ms` 等。

### 3. 得到延迟后怎么用

- 把延迟填回 `config/task_config.yaml` 里 `output_train_data` 的 `visual_cam_latency` / `tactile_cam_latency` 等。
- 跑 **02_cali_offset_latency** 时，会用这些值去对齐轨迹和视频时间戳。

---

## 四、对照小结

| 项目       | 手眼标定                     | 延迟标定                         |
|------------|------------------------------|----------------------------------|
| ArUco 来源 | 打印的 ID 10 大码（场景中）  | 屏幕显示（vis_aruco），无需打印  |
| 你的配合   | 录视觉视频 + Quest 位姿      | 先 vis_aruco，再相机录屏         |
| 脚本       | `cali_camera_quest_hand.py`  | `vis_aruco` → `get_cali_latency_video` → `cali_cam_latency` |
| 输出       | `T_CameraToQuestHand.npy`    | `latency_result.json`            |

你已有 ID 10 大码，手眼标定直接用即可；延迟标定按上面三步跑现有代码就够了。
