import cv2
import numpy as np
import os
from pathlib import Path
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from datetime import datetime

class ImageProcessor:
    """离线处理图片文件夹的运动检测器"""
    
    def __init__(self, angle_threshold=15, cluster_threshold=10, 
                 min_threshold=10, gray_threshold=0.46, dbscan_eps=5, dbscan_min_samples=3):
        
        self.angle_threshold = angle_threshold
        self.cluster_threshold = cluster_threshold
        self.min_threshold = min_threshold
        self.gray_threshold = gray_threshold
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        
        # 缓存三帧数据
        self.pre = {'frame': None, 'ob': None, 'timestamp': None}
        self.last = {'frame': None, 'ob': None, 'timestamp': None}
        self.next = {'frame': None, 'ob': None, 'timestamp': None}
        
        # 中间结果存储
        self.mask_labels = None
        self.mask_stats = None
        self.mask_centroids = None
        self.edge_mask = []
        self.fin_edge_p = []
        self.fin_mask = []
        self.flow = None
        self.next_target_id = 0
        
        # 颜色映射表（用于区分不同物体）
        self.color_map = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0),
            (128, 128, 0), (0, 128, 128), (128, 0, 0), (0, 128, 0)
        ]
    
    def _id_(self):
        self.next_target_id += 1
        return self.next_target_id
    
    def _gray_image_(self, img):
        """灰度化、归一化、阈值过滤"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray[gray < self.gray_threshold] = 0
        return gray
    
    def _angle_to_color(self, angle):
        """角度映射到色盘（HSV色相）"""
        hue = angle / 360.0
        hsv = np.array([[[hue * 180, 255, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return (int(bgr[0,0,0]), int(bgr[0,0,1]), int(bgr[0,0,2]))
    
    def _velocity_to_color(self, velocity, max_vel=10.0):
        """速度映射到明度"""
        v_norm = min(velocity / max_vel, 1.0)
        value = int(128 + v_norm * 127)
        return (value, value, value)
    
    def _connected(self, img):
        """连通域分析，保存labels数组"""
        binary = (img > 0).astype(np.uint8) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        self.mask_labels = labels
        self.mask_stats = stats
        self.mask_centroids = centroids
        return labels, stats, centroids
    
    def _to_edge(self):
        """边缘提取"""
        self.edge_mask = []
        kernel = np.ones((5, 5), np.uint8)
        
        if self.mask_labels is None:
            return
        
        unique_labels = np.unique(self.mask_labels)
        for label in unique_labels:
            if label == 0:
                continue
            mask = (self.mask_labels == label).astype(np.uint8) * 255
            edge_mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
            self.edge_mask.append(edge_mask)
    
    def _flow(self, last_img, next_img):
        """稠密光流计算"""
        last_uint8 = (last_img * 255).astype(np.uint8)
        next_uint8 = (next_img * 255).astype(np.uint8)
        
        # 帧差法获取运动区域
        diff = cv2.absdiff(last_uint8, next_uint8)
        _, diff_binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        non_zero_points = np.column_stack(np.where(diff_binary > 0))
        
        if len(non_zero_points) == 0:
            # 全图光流
            flow = cv2.calcOpticalFlowFarneback(
                last_uint8, next_uint8, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.1, flags=0
            )
            self.flow = flow
            return
        
        # 裁剪运动区域
        y_min, x_min = non_zero_points.min(axis=0)
        y_max, x_max = non_zero_points.max(axis=0)
        margin = 10
        y_min = max(0, y_min - margin)
        x_min = max(0, x_min - margin)
        y_max = min(last_uint8.shape[0], y_max + margin)
        x_max = min(last_uint8.shape[1], x_max + margin)
        
        re_last = last_uint8[y_min:y_max, x_min:x_max]
        re_next = next_uint8[y_min:y_max, x_min:x_max]
        
        flow_roi = cv2.calcOpticalFlowFarneback(
            re_last, re_next, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.1, flags=0
        )
        
        h, w = last_uint8.shape
        self.flow = np.zeros((h, w, 2), dtype=np.float32)
        self.flow[y_min:y_max, x_min:x_max] = flow_roi
    
    def _to_flaw(self):
        """DBSCAN聚类分割边缘点"""
        self.fin_edge_p = []
        
        if self.flow is None:
            return
        
        for edge_mask in self.edge_mask:
            ys, xs = np.where(edge_mask > 0)
            if len(ys) < 3:
                continue
            
            # 提取速度矢量
            v_flow = []
            for y, x in zip(ys, xs):
                vx = self.flow[y, x, 0]
                vy = self.flow[y, x, 1]
                v_flow.append([vx, vy])
            v_flow = np.array(v_flow)
            
            # DBSCAN聚类
            clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(v_flow)
            labels = clustering.labels_
            
            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1:
                    continue
                mask_idx = np.where(labels == label)[0]
                edge_p = np.column_stack([xs[mask_idx], ys[mask_idx]])
                if len(edge_p) >= 3:
                    self.fin_edge_p.append(edge_p)
    
    def _to_mask(self):
        """生成最终目标掩码和OB信息"""
        self.last["ob"] = {}
        self.fin_mask = []
        
        if self.mask_labels is None:
            return
        
        h, w = self.last['frame'].shape
        combined_orig_mask = (self.mask_labels > 0).astype(np.uint8)
        
        for edge_points in self.fin_edge_p:
            if len(edge_points) < 3:
                continue
            
            points_float = edge_points.astype(np.float32)
            (rx, ry), r = cv2.minEnclosingCircle(points_float)
            rx, ry, r = int(round(rx)), int(round(ry)), int(round(r))
            
            cir = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(cir, (rx, ry), r, 1, -1)
            intersection = cv2.bitwise_and(combined_orig_mask, cir)
            
            if np.sum(intersection) < 10:
                continue
            
            points_final = np.column_stack(np.where(intersection > 0))
            if len(points_final) < 3:
                continue
            
            (cx, cy), cr = cv2.minEnclosingCircle(points_final.astype(np.float32))
            cx, cy, cr = int(round(cx)), int(round(cy)), int(round(cr))
            
            # 保存最终掩码
            final_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(final_mask, (cx, cy), cr, 1, -1)
            self.fin_mask.append(final_mask)
            
            # 计算平均方向
            dir_angles = []
            vel_values = []
            if self.flow is not None:
                for y, x in points_final:
                    vx = self.flow[y, x, 0]
                    vy = self.flow[y, x, 1]
                    if abs(vx) > 0.01 or abs(vy) > 0.01:
                        angle = np.arctan2(vy, vx) * 180 / np.pi
                        if angle < 0:
                            angle += 360
                        dir_angles.append(angle)
                        vel_values.append(np.sqrt(vx**2 + vy**2))
            
            dir_mean = np.mean(dir_angles) if dir_angles else 0.0
            v_mean = np.mean(vel_values) if vel_values else 0.0
            
            ob_id = self._id_()
            self.last["ob"][ob_id] = {
                "center": (cx, cy),
                "radius": cr,
                "direction": dir_mean,
                "velocity": v_mean
            }
    
    def process_frame_pair(self, idx, pre_img_bgr, last_img_bgr, next_img_bgr, 
                           pre_timestamp, last_timestamp, next_timestamp, save_dir):
        """处理一组三帧图片"""
        
        print(f"\n处理第 {idx} 轮: {pre_timestamp} -> {last_timestamp} -> {next_timestamp}")
        
        # 灰度预处理
        pre_gray = self._gray_image_(pre_img_bgr)
        last_gray = self._gray_image_(last_img_bgr)
        next_gray = self._gray_image_(next_img_bgr)
        
        # 保存到缓存
        self.pre['frame'] = pre_gray
        self.last['frame'] = last_gray
        self.next['frame'] = next_gray
        self.pre['timestamp'] = pre_timestamp
        self.last['timestamp'] = last_timestamp
        self.next['timestamp'] = next_timestamp
        
        # 执行处理流程
        self._connected(last_gray)
        self._to_edge()
        self._flow(last_gray, next_gray)
        self._to_flaw()
        self._to_mask()
        
        h, w = last_gray.shape
        
        # 1. 保存 mask 图像（不同物体不同颜色）
        mask_color = np.zeros((h, w, 3), dtype=np.uint8)
        if self.mask_labels is not None:
            unique_labels = np.unique(self.mask_labels)
            color_idx = 0
            for label in unique_labels:
                if label == 0:
                    continue
                color = self.color_map[color_idx % len(self.color_map)]
                mask_color[self.mask_labels == label] = color
                color_idx += 1
        cv2.imwrite(str(save_dir / f"round_{idx:04d}_mask.jpg"), mask_color)
        
        # 2. 保存 edge_mask 图像（不同物体不同颜色）
        edge_color = np.zeros((h, w, 3), dtype=np.uint8)
        for i, edge in enumerate(self.edge_mask):
            color = self.color_map[i % len(self.color_map)]
            edge_color[edge > 0] = color
        cv2.imwrite(str(save_dir / f"round_{idx:04d}_edge_mask.jpg"), edge_color)
        
        # 3. 保存 fin_edge_p 图像（不同物体不同颜色）
        fin_edge_color = np.zeros((h, w, 3), dtype=np.uint8)
        for i, points in enumerate(self.fin_edge_p):
            color = self.color_map[i % len(self.color_map)]
            for x, y in points:
                if 0 <= x < w and 0 <= y < h:
                    fin_edge_color[y, x] = color
        cv2.imwrite(str(save_dir / f"round_{idx:04d}_fin_edge.jpg"), fin_edge_color)
        
        # 4. 保存 fin_mask 图像（不同物体不同颜色）
        fin_mask_color = np.zeros((h, w, 3), dtype=np.uint8)
        for i, fm in enumerate(self.fin_mask):
            color = self.color_map[i % len(self.color_map)]
            fin_mask_color[fm > 0] = color
        cv2.imwrite(str(save_dir / f"round_{idx:04d}_fin_mask.jpg"), fin_mask_color)
        
        # 5. 光流方向图（角度映射到色盘）
        if self.flow is not None:
            direct_color = np.zeros((h, w, 3), dtype=np.uint8)
            for y in range(h):
                for x in range(w):
                    vx, vy = self.flow[y, x, 0], self.flow[y, x, 1]
                    if abs(vx) > 0.01 or abs(vy) > 0.01:
                        angle = np.arctan2(vy, vx) * 180 / np.pi
                        if angle < 0:
                            angle += 360
                        direct_color[y, x] = self._angle_to_color(angle)
            cv2.imwrite(str(save_dir / f"round_{idx:04d}_light_direct.jpg"), direct_color)
        
        # 6. 光流速度图（速度映射到明度）
        if self.flow is not None:
            v_color = np.zeros((h, w, 3), dtype=np.uint8)
            max_v = 5.0
            for y in range(h):
                for x in range(w):
                    v_val = np.sqrt(self.flow[y, x, 0]**2 + self.flow[y, x, 1]**2)
                    if v_val > 0.01:
                        v_color[y, x] = self._velocity_to_color(v_val, max_v)
            cv2.imwrite(str(save_dir / f"round_{idx:04d}_light_v.jpg"), v_color)
        
        # 7. 原图标记结果
        result_img = last_img_bgr.copy()
        for ob_id, ob_info in self.last["ob"].items():
            cx, cy = ob_info["center"]
            r = ob_info["radius"]
            direction = ob_info["direction"]
            velocity = ob_info["velocity"]
            
            cv2.circle(result_img, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(result_img, (cx, cy), 3, (0, 0, 255), -1)
            label = f"ID:{ob_id} dir:{direction:.1f} vel:{velocity:.1f}"
            cv2.putText(result_img, label, (cx - r, cy - r - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            rad = direction * np.pi / 180
            end_x = int(cx + r * 0.8 * np.cos(rad))
            end_y = int(cy + r * 0.8 * np.sin(rad))
            cv2.arrowedLine(result_img, (cx, cy), (end_x, end_y), (0, 255, 255), 2)
        
        cv2.imwrite(str(save_dir / f"round_{idx:04d}_result.jpg"), result_img)
        
        print(f"  识别到 {len(self.last['ob'])} 个物体")
        return len(self.last['ob'])
    
    def run(self, input_folder, output_folder):
        """运行完整处理流程"""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有PNG图片，按文件名排序
        png_files = sorted(input_path.glob("*.png"))
        print(f"找到 {len(png_files)} 张图片")
        
        if len(png_files) < 3:
            print("图片数量不足3张，无法处理")
            return
        
        # 逐组处理
        for idx in range(len(png_files) - 2):
            pre_img = cv2.imread(str(png_files[idx]))
            last_img = cv2.imread(str(png_files[idx + 1]))
            next_img = cv2.imread(str(png_files[idx + 2]))
            
            pre_ts = png_files[idx].stem
            last_ts = png_files[idx + 1].stem
            next_ts = png_files[idx + 2].stem
            
            # 重置ID
            self.next_target_id = 0
            
            self.process_frame_pair(idx + 1, pre_img, last_img, next_img,
                                     pre_ts, last_ts, next_ts, output_path)
        
        print(f"\n处理完成！结果保存在 {output_path}")


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    input_folder = "C:/mypy/recog_ai/imgs/flaw"
    output_folder = "C:/mypy/recog_ai/imgs/flaw/result"
    
    processor = ImageProcessor(
        gray_threshold=0.46,
        dbscan_eps=5,
        dbscan_min_samples=3
    )
    
    processor.run(input_folder, output_folder)