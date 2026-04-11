import cv2
import numpy as np
import os
from pathlib import Path
from sklearn.cluster import DBSCAN

class MotionAnalyzer:
    def __init__(self, gray_threshold=0.5, angle_threshold=15, distance_threshold=15):
        """
        初始化运动分析器
        gray_threshold: 灰度阈值，低于此值的像素被忽略
        angle_threshold: 角度聚类阈值（度），同一物体内角度差不能超过此值
        distance_threshold: 位置聚类阈值（像素），同一物体内点距离不能超过此值
        """
        self.gray_threshold = gray_threshold
        self.angle_threshold = angle_threshold
        self.distance_threshold = distance_threshold
        self.pre_frame = None
        self.last_frame = None
        self.last_light_direct = None
        self.frame_list = []
        self.timestamp_list = []

    def load_frames_from_folder(self, folder_path):
        """
        从文件夹加载按时间戳命名的PNG图片，按文件名排序
        输入：文件夹路径
        输出：填充 self.frame_list 和 self.timestamp_list
        """
        png_files = sorted(Path(folder_path).glob("*.png"))
        for png_path in png_files:
            # 文件名（不含扩展名）作为时间戳
            timestamp = png_path.stem
            img = cv2.imread(str(png_path))
            if img is not None:
                self.frame_list.append(img)
                self.timestamp_list.append(timestamp)
                print(f"加载图片: {png_path.name} -> 时间戳: {timestamp}")
        print(f"共加载 {len(self.frame_list)} 张图片")

    def _gray_image(self, img, ignore_xy=None):
        """
        图像预处理：灰度化、归一化、阈值过滤
        输入：img(BGR三通道), ignore_xy(忽略区域坐标 [x1,y1,x2,y2])
        输出：processed_img(灰度图 float32 0-1)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray[gray < self.gray_threshold] = 0

        if ignore_xy and len(ignore_xy) == 4:
            x1, y1, x2, y2 = [int(round(v)) for v in ignore_xy]
            w = x2 - x1
            h = y2 - y1
            if w > 0 and h > 0:
                gray[y1:y1+h, x1:x1+w] = 0

        return gray

    def _compute_flow(self, pre_img, last_img):
        """
        计算光流场（从 pre_img 到 last_img）
        输入：两帧灰度图 (H, W) float32 0-1
        输出：光流场 (H, W, 2) 每个像素存储 (vx, vy)
        """
        # 转换为 uint8 格式供 OpenCV 光流使用
        pre_uint8 = (pre_img * 255).astype(np.uint8)
        last_uint8 = (last_img * 255).astype(np.uint8)
        
        flow = cv2.calcOpticalFlowFarneback(
            pre_uint8, last_uint8, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        return flow

    def _connected_components(self, img):
        """
        对二值图像做连通域分析，分离不同物体
        输入：img (H, W) float32 0-1
        输出：掩码列表，每个掩码是二值图 (H, W) uint8 0/255
        """
        # 二值化：非零像素作为前景
        binary = (img > 0).astype(np.uint8) * 255
        
        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        masks = []
        for i in range(1, num_labels):  # 跳过背景（标签0）
            # 创建单个物体的掩码
            mask = (labels == i).astype(np.uint8) * 255
            # 过滤面积太小的区域（小于10像素）
            if stats[i, cv2.CC_STAT_AREA] >= 10:
                masks.append(mask)
        
        return masks

    def _to_edge(self, mask):
        """
        掩码边缘提取，目标以边缘为依据
        输入：单个掩码 (H, W) uint8 0/255
        输出：边缘掩码 (H, W) uint8 0/255
        """
        kernel = np.ones((5, 5), np.uint8)
        edge_mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        return edge_mask

    def _extract_angle_points(self, edge_mask, flow):
        """
        从边缘掩码中提取像素，配合光流图对应点角度
        输入：edge_mask (H, W) uint8, flow (H, W, 2)
        输出：点列表 [(角度, x, y, vx, vy), ...]
        """
        # 找到边缘掩码中非零的位置
        ys, xs = np.where(edge_mask > 0)
        if len(ys) == 0:
            return []
        
        # 批量获取光流矢量
        vx_vals = flow[ys, xs, 0]
        vy_vals = flow[ys, xs, 1]
        
        # 批量计算角度（度数，0-360）
        angles = np.arctan2(vy_vals, vx_vals) * 180 / np.pi
        angles[angles < 0] += 360
        
        # 组合成点列表
        points = []
        for i in range(len(ys)):
            points.append((angles[i], xs[i], ys[i], vx_vals[i], vy_vals[i]))
        
        return points

    def _angle_clustering(self, points):
        """
        按运动方向角度进行聚类
        输入：点列表 [(角度, x, y, vx, vy), ...]
        输出：角度聚类后的列表，每个元素是一个点列表
        """
        if len(points) == 0:
            return []
        
        # 按角度排序
        points.sort(key=lambda p: p[0])
        
        angle_diff = lambda a, b: min(abs(a - b), 360 - abs(a - b))
        
        clusters = []
        cur_cluster = [points[0]]
        
        for i in range(1, len(points)):
            diff = angle_diff(points[i][0], points[i-1][0])
            if diff > self.angle_threshold:
                clusters.append(cur_cluster)
                cur_cluster = [points[i]]
            else:
                cur_cluster.append(points[i])
        
        if cur_cluster:
            clusters.append(cur_cluster)
        
        return clusters

    def _position_clustering(self, angle_clusters):
        """
        在每个方向组内按空间欧氏距离聚类
        输入：角度聚类后的列表
        输出：最终物体点簇列表
        """
        all_objects = []
        
        for group in angle_clusters:
            if len(group) < 3:
                continue
            
            # 按 x 坐标排序
            group.sort(key=lambda p: (p[1], p[2]))
            
            objects_in_group = []
            cur_object = [group[0]]
            
            for i in range(1, len(group)):
                dx = group[i][1] - group[i-1][1]
                dy = group[i][2] - group[i-1][2]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist > self.distance_threshold:
                    if len(cur_object) >= 3:
                        objects_in_group.append(cur_object)
                    cur_object = [group[i]]
                else:
                    cur_object.append(group[i])
            
            if len(cur_object) >= 3:
                objects_in_group.append(cur_object)
            
            all_objects.extend(objects_in_group)
        
        return all_objects

    def _angle_to_color(self, angle):
        """
        将角度映射到色盘（HSV色相），返回BGR颜色
        输入：角度 0-360
        输出：BGR颜色元组 (b, g, r)
        """
        # 色相：角度直接映射（0°=红色，120°=绿色，240°=蓝色）
        hue = angle / 360.0
        # 固定饱和度和明度为最大值
        hsv = np.array([[[hue * 180, 255, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return (int(bgr[0,0,0]), int(bgr[0,0,1]), int(bgr[0,0,2]))

    def process_frame_pair(self, idx):
        """
        处理一对相邻帧
        输入：帧索引（当前帧作为 last，上一帧作为 pre）
        输出：是否处理成功
        """
        if idx == 0:
            print(f"第 {idx} 帧没有上一帧，跳过")
            return False
        
        pre_img_bgr = self.frame_list[idx-1]
        last_img_bgr = self.frame_list[idx]
        timestamp = self.timestamp_list[idx]
        prev_timestamp = self.timestamp_list[idx-1]
        
        print(f"\n处理第 {idx} 对: {prev_timestamp} -> {timestamp}")
        
        # 1. 灰度预处理
        pre_gray = self._gray_image(pre_img_bgr)
        last_gray = self._gray_image(last_img_bgr)
        
        # 2. 计算光流
        flow = self._compute_flow(pre_gray, last_gray)
        
        # 3. 对 last 帧做连通域分析
        masks = self._connected_components(last_gray)
        print(f"  连通域分析得到 {len(masks)} 个初始物体")
        
        # 4. 对每个掩码提取边缘，收集边缘点及其角度
        all_points = []
        edge_masks = []
        
        for mask in masks:
            edge_mask = self._to_edge(mask)
            edge_masks.append(edge_mask)
            points = self._extract_angle_points(edge_mask, flow)
            all_points.extend(points)
        
        print(f"  提取到 {len(all_points)} 个边缘运动点")
        
        # 5. 角度聚类 + 位置聚类
        angle_clusters = self._angle_clustering(all_points)
        object_clusters = self._position_clustering(angle_clusters)
        print(f"  最终识别到 {len(object_clusters)} 个运动物体")
        
        # 6. 生成结果文件
        result_folder = Path("C:/mypy/recog_ai/imgs/flaw/result")
        result_folder.mkdir(parents=True, exist_ok=True)
        
        # 6a. 保存角度文本文件
        txt_path = result_folder / f"round_{idx}_angles.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            for obj_idx, cluster in enumerate(object_clusters):
                f.write(f"物体 {obj_idx + 1}:\n")
                angles_str = ", ".join([f"{p[0]:.2f}" for p in cluster])
                f.write(angles_str + "\n\n")
        print(f"  保存角度文件: {txt_path}")
        
        # 6b. 生成并保存色盘可视化图
        # 创建空白彩色图
        h, w = last_gray.shape
        color_vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 为每个物体的每个边缘点着色
        for cluster in object_clusters:
            for point in cluster:
                angle, x, y = point[0], point[1], point[2]
                color = self._angle_to_color(angle)
                color_vis[y, x] = color
        
        # 保存
        vis_path = result_folder / f"round_{idx}_visual.jpg"
        cv2.imwrite(str(vis_path), color_vis)
        print(f"  保存可视化图: {vis_path}")
        
        # 可选：保存边缘掩码叠加图
        edge_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for edge_mask in edge_masks:
            edge_overlay[edge_mask > 0] = [0, 255, 0]  # 绿色边缘
        overlay_path = result_folder / f"round_{idx}_edges.jpg"
        cv2.imwrite(str(overlay_path), edge_overlay)
        
        return True

    def run(self, folder_path):
        """
        运行完整流程
        """
        # 1. 加载所有图片
        self.load_frames_from_folder(folder_path)
        
        if len(self.frame_list) < 2:
            print("图片数量不足2张，无法进行光流计算")
            return
        
        # 2. 逐对处理相邻帧
        for idx in range(1, len(self.frame_list)):
            self.process_frame_pair(idx)
        
        print("\n处理完成！")


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = MotionAnalyzer(
        gray_threshold=0.5,      # 灰度阈值
        angle_threshold=15,      # 角度聚类阈值（度）
        distance_threshold=15    # 位置聚类阈值（像素）
    )
    
    # 运行分析
    analyzer.run("C:/mypy/recog_ai/imgs/flaw")