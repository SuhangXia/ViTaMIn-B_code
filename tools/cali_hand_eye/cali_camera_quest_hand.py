#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手眼标定：仅用 ArUco 码 + Quest 手柄，不依赖机械臂，求 T_CameraToQuestHand。

AX = XB 形式（与图中一致）：
  - A_i = T_QuestHandToHS_i（手柄→头显，即 gripper→base）
  - B_i = T_ArucoToCamera_i（ArUco→相机，即 target→camera）
  - X = T_CameraToQuestHand（待求）

数据：固定 ArUco 在场景中，移动带相机的手柄；或固定相机，移动带 ArUco 的手柄。
同时录制视觉视频 + Quest 位姿（00_get_pose），保证时间重叠。

支持：1）--video + 内参 + aruco_yaml：对视频逐帧跑 ArUco；2）--aruco_pkl：用已有 tag_detection_*.pkl。
求解：--method opencv（默认）或 svd。
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# 项目根目录
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.cv_util import (
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)


def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """从 solvePnP 的 rvec/tvec 得到 4x4 T_CameraToAruco。"""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def quest_wrist_to_T(wrist: dict) -> np.ndarray:
    """Quest JSON 的 left_wrist / right_wrist -> 4x4 T_HSToQuestHand（头显系下手柄位姿）。"""
    p = wrist["position"]
    r = wrist["rotation"]
    pos = np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
    quat_xyzw = np.array([r["x"], r["y"], r["z"], r["w"]], dtype=np.float64)
    rot = Rotation.from_quat(quat_xyzw)
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = pos
    return T


def load_quest_poses(
    quest_dir: Path | None,
    hand: str,
    quest_jsons: list[Path] | None = None,
) -> list[tuple[float, np.ndarray]]:
    """加载 Quest 位姿：(timestamp, T_HSToQuestHand) 列表，按时间排序。"""
    files: list[Path] = []
    if quest_jsons:
        files = [Path(p) for p in quest_jsons]
    elif quest_dir is not None:
        d = Path(quest_dir)
        if not d.exists():
            raise FileNotFoundError(f"目录不存在: {d}。请先创建并采集 Quest 位姿（00_get_pose）到 data/<任务>/all_trajectory/。")
        files = sorted(d.rglob("quest_poses_*.json"))
        if not files and (d / "all_trajectory").is_dir():
            files = sorted((d / "all_trajectory").glob("quest_poses_*.json"))
    if not files:
        hint = (
            "请先采集 Quest 位姿（如 00_get_pose --cfg config/task_config.yaml），"
            "保存到 data/<任务>/all_trajectory/ 下的 quest_poses_*.json；"
            "或指定包含这些文件的目录 --quest_dir，或显式传 --quest_jsons。"
        )
        loc = str(quest_dir) if quest_dir else "未指定 quest_dir"
        raise FileNotFoundError(f"未找到 quest_poses_*.json（{loc}）。{hint}")

    key = "left_wrist" if hand == "left" else "right_wrist"
    out = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        for frame in data:
            if key not in frame:
                continue
            t = float(frame["timestamp_unix"]) if "timestamp_unix" in frame else float(frame.get("timestamp", 0.0))
            w = frame[key]
            T = quest_wrist_to_T(w)
            out.append((t, T))

    out.sort(key=lambda x: x[0])
    return out


def load_aruco_from_video(
    video_path: Path,
    intrinsics_json: Path,
    aruco_yaml: Path,
    aruco_id: int,
) -> list[tuple[float, np.ndarray]]:
    """对视频逐帧做 ArUco 检测，返回 (time_sec, T_CameraToAruco) 列表。"""
    import av

    aruco_config = parse_aruco_config(yaml.safe_load(open(aruco_yaml, "r")))
    raw_intr = parse_fisheye_intrinsics(json.load(open(intrinsics_json, "r")))
    results = []

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        tb = stream.time_base
        res = (stream.width, stream.height)
        intr = convert_fisheye_intrinsics_resolution(raw_intr, res)

        for i, frame in enumerate(tqdm(container.decode(stream), desc="ArUco on video")):
            img = frame.to_ndarray(format="rgb24")
            t = frame.pts * tb if frame.pts is not None else i / fps
            tag_dict = detect_localize_aruco_tags(
                img,
                aruco_dict=aruco_config["aruco_dict"],
                marker_size_map=aruco_config["marker_size_map"],
                fisheye_intr_dict=intr,
                refine_subpix=True,
            )
            if aruco_id not in tag_dict:
                continue
            rvec = tag_dict[aruco_id]["rvec"]
            tvec = tag_dict[aruco_id]["tvec"]
            T = rvec_tvec_to_T(rvec, tvec)
            results.append((float(t), T))

    return results


def load_aruco_from_pkl(pkl_path: Path, aruco_id: int) -> list[tuple[float, np.ndarray]]:
    """从 detect_aruco 的 pkl 加载 (time_sec, T_CameraToAruco)。"""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    out = []
    for rec in data:
        td = rec.get("tag_dict") or {}
        if aruco_id not in td:
            continue
        t = float(rec.get("time", 0.0))
        rvec = td[aruco_id]["rvec"]
        tvec = td[aruco_id]["tvec"]
        T = rvec_tvec_to_T(rvec, tvec)
        out.append((t, T))
    out.sort(key=lambda x: x[0])
    return out


def sync_pairs(
    aruco_list: list[tuple[float, np.ndarray]],
    quest_list: list[tuple[float, np.ndarray]],
    max_dt: float,
    quest_time_offset: float = 0.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """对每个 ArUco 时刻找最近 Quest 时刻，保留 dt < max_dt 的配对。返回 (A_i, B_i) 列表。"""
    q_ts = np.array([x[0] for x in quest_list])
    q_Ts = [x[1] for x in quest_list]
    pairs = []
    for t_a, T_cam2aru in aruco_list:
        t_q = t_a + quest_time_offset
        idx = np.argmin(np.abs(q_ts - t_q))
        dt = abs(q_ts[idx] - t_q)
        if dt > max_dt:
            continue
        T_hs2hand = q_Ts[idx]
        A = np.linalg.inv(T_hs2hand)  # T_QuestHandToHS (gripper->base)
        B = np.linalg.inv(T_cam2aru)  # T_ArucoToCamera (target->cam)
        pairs.append((A, B))
    return pairs


def solve_hand_eye_opencv(pairs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """OpenCV calibrateHandEye：返回 4x4 T_CameraToQuestHand。"""
    R_g2b = [p[0][:3, :3].astype(np.float64) for p in pairs]
    t_g2b = [p[0][:3, 3].astype(np.float64).reshape(3, 1) for p in pairs]
    R_t2c = [p[1][:3, :3].astype(np.float64) for p in pairs]
    t_t2c = [p[1][:3, 3].astype(np.float64).reshape(3, 1) for p in pairs]

    R, t = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_DANIILIDIS
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    return T


def solve_hand_eye_svd(pairs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """SVD 最小二乘求解 AX=XB，返回 4x4 T_CameraToQuestHand。"""
    # 旋转： stack (R_A_i - R_B_i) 的线性化形式，求 R_X
    # 使用 kronecker: (I ⊗ R_A - R_B^T ⊗ I) vec(R_X) = 0
    M_rot = []
    for A, B in pairs:
        Ra, Rb = A[:3, :3], B[:3, :3]
        M_rot.append(np.kron(np.eye(3), Ra) - np.kron(Rb.T, np.eye(3)))
    M = np.vstack(M_rot)
    _, _, Vt = np.linalg.svd(M)
    # 最小奇异向量对应 vec(R_X)，reshape 为 3x3
    v = Vt[-1]
    R_x = v.reshape(3, 3)
    # 投影到 SO(3)
    U, _, Vt = np.linalg.svd(R_x)
    R_x = U @ Vt
    if np.linalg.det(R_x) < 0:
        Vt[-1] *= -1
        R_x = U @ Vt

    # 平移：A X = X B => (R_A - I) t_X = R_X t_B - t_A
    C = []
    d = []
    for A, B in pairs:
        Ra, ta = A[:3, :3], A[:3, 3]
        tb = B[:3, 3]
        C.append(Ra - np.eye(3))
        d.append(R_x @ tb - ta)
    C = np.vstack(C)
    d = np.concatenate(d)
    t_x, _, _, _ = np.linalg.lstsq(C, d, rcond=None)

    T = np.eye(4)
    T[:3, :3] = R_x
    T[:3, 3] = t_x
    return T


def main():
    ap = argparse.ArgumentParser(description="ArUco + Quest 手眼标定，求 T_CameraToQuestHand")
    ap.add_argument("--quest_dir", type=Path, default=None, help="Quest JSON 所在目录；会递归及 all_trajectory/ 查找 quest_poses_*.json")
    ap.add_argument("--quest_jsons", type=Path, nargs="*", default=None, help="直接指定 quest_poses_*.json 文件路径（可多个）")
    ap.add_argument("--hand", choices=["left", "right"], default="right", help="左手 / 右手")
    ap.add_argument("--aruco_id", type=int, default=0, help="使用的 ArUco marker ID")
    ap.add_argument("--max_sync_error_sec", type=float, default=0.05, help="同步最大时间差（秒）")
    ap.add_argument("--quest_time_offset", type=float, default=0.0, help="Quest 时间整体偏移（秒）；若与 --search_time_offset 同用则作为初值")
    ap.add_argument("--search_time_offset", action="store_true", help="在 ±1s 内搜索最佳 Quest 时间偏移（最大化有效配对）")
    ap.add_argument("--method", choices=["opencv", "svd"], default="opencv", help="求解方法")
    ap.add_argument("--output", type=Path, default=Path("T_CameraToQuestHand.npy"), help="输出 4x4 矩阵路径")

    g = ap.add_argument_group("ArUco 数据来源（二选一）")
    g.add_argument("--video", type=Path, help="标定视频路径（与 Quest 同次录制）")
    g.add_argument("--aruco_pkl", type=Path, help="已有 ArUco 检测 pkl（tag_detection_*.pkl）")

    g2 = ap.add_argument_group("使用 --video 时必填")
    g2.add_argument("--intrinsics_json", type=Path, help="相机内参 JSON")
    g2.add_argument("--aruco_yaml", type=Path, help="ArUco 配置 YAML")

    args = ap.parse_args()

    if args.video is None and args.aruco_pkl is None:
        ap.error("请指定 --video 或 --aruco_pkl")
    if args.video is not None and (args.intrinsics_json is None or args.aruco_yaml is None):
        ap.error("使用 --video 时请同时指定 --intrinsics_json 和 --aruco_yaml")
    if not args.quest_jsons and not args.quest_dir:
        ap.error("请指定 --quest_dir 或 --quest_jsons")

    quest_list = load_quest_poses(args.quest_dir, args.hand, args.quest_jsons)
    print(f"[Quest] 加载 {len(quest_list)} 条位姿")

    if args.aruco_pkl is not None:
        aruco_list = load_aruco_from_pkl(args.aruco_pkl, args.aruco_id)
        print(f"[ArUco] 从 pkl 加载 {len(aruco_list)} 条（aruco_id={args.aruco_id}）")
    else:
        aruco_list = load_aruco_from_video(
            args.video,
            args.intrinsics_json,
            args.aruco_yaml,
            args.aruco_id,
        )
        print(f"[ArUco] 从视频检测得到 {len(aruco_list)} 条（aruco_id={args.aruco_id}）")

    offset = args.quest_time_offset
    if args.search_time_offset:
        best_n, best_offset = 0, offset
        for o in np.arange(offset - 1.0, offset + 1.0 + 1e-6, 0.02):
            p = sync_pairs(aruco_list, quest_list, args.max_sync_error_sec, quest_time_offset=o)
            if len(p) > best_n:
                best_n, best_offset = len(p), o
        offset = best_offset
        print(f"[Sync] 时间偏移搜索得到 best offset={best_offset:.3f}s, 配对 {best_n} 条")
    pairs = sync_pairs(
        aruco_list,
        quest_list,
        max_dt=args.max_sync_error_sec,
        quest_time_offset=offset,
    )
    print(f"[Sync] 有效配对 {len(pairs)} 条（max_dt={args.max_sync_error_sec}s, offset={offset:.3f}s）")

    if len(pairs) < 4:
        print("[ERROR] 至少需要 4 对有效数据，请增加标定采集或放宽 --max_sync_error_sec")
        sys.exit(1)

    if args.method == "opencv":
        T = solve_hand_eye_opencv(pairs)
    else:
        T = solve_hand_eye_svd(pairs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, T)
    print(f"\nT_CameraToQuestHand 已保存: {args.output}")
    print(T)


# 示例：详见 tools/CALIBRATION_USAGE.md
#   --quest_dir 填 data/<任务> 或含 quest_poses_*.json 的目录；或 --quest_jsons 直接指定 json。
#   --aruco_id 10 对应你现有大码；yaml 里 marker_size_map 的 10 要改成实际尺寸（米）。

if __name__ == "__main__":
    main()
