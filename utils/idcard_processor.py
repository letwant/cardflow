import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO
from typing import Dict, Optional


class IDCardProcessor:
    def __init__(self,
                 seg_model_path: str = "models/yolov8-idcard-seg.pt",
                 det_model_path: str = "models/yolov8-idcard-detect.pt",
                 save_dir: str = "static/uploads",
                 device: str = "cpu"):
        """
        初始化身份证处理器
        :param seg_model_path: 分割模型路径
        :param det_model_path: 检测模型路径
        :param save_dir: 结果保存目录
        :param device: 计算设备(cpu/0/1等)
        """
        self.seg_model = YOLO(seg_model_path)
        self.det_model = YOLO(det_model_path)
        self.save_dir = save_dir
        self.device = device
        os.makedirs(save_dir, exist_ok=True)

        # 验证设备可用性
        if "cuda" in device and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available")

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """顶点排序（私有方法）"""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _correct_perspective(self, img: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """图像透视矫正（私有方法）"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        max_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)

        # 轮廓优化逻辑
        if len(approx) < 4:
            hull = cv2.convexHull(max_contour)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            if len(approx) < 4:
                return None

        ordered_pts = self._order_points(approx[:, 0, :])

        # 动态计算目标尺寸
        width = np.mean([np.linalg.norm(ordered_pts[0] - ordered_pts[1]),
                         np.linalg.norm(ordered_pts[2] - ordered_pts[3])])
        height = np.mean([np.linalg.norm(ordered_pts[1] - ordered_pts[2]),
                          np.linalg.norm(ordered_pts[3] - ordered_pts[0])])

        # 自适应方向判断
        if height > width:
            target_size = (int(height), int(height / 1.586))
            ordered_pts = np.roll(ordered_pts, -1, axis=0)
        else:
            target_size = (int(width), int(width / 1.586))

        dst_pts = np.array([[0, 0], [target_size[0], 0],
                            [target_size[0], target_size[1]], [0, target_size[1]]], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
        return cv2.warpPerspective(img, matrix, target_size)

    def _detect_orientation(self, img: np.ndarray) -> bool:
        """方向检测（私有方法）
        逻辑：
            1. 优先选择 label 为 "face" 或 "emblem" 的 box。
            2. 如果 label 是 "face"，判断其中心点是否在图像中心点左侧。
            3. 如果 label 是 "emblem"，判断其中心点是否在图像中心点右侧。
        """
        results = self.det_model.predict(img, conf=0.2, verbose=False)

        for r in results:
            if r.boxes:
                h, w = img.shape[:2]
                image_center_x = w // 2  # 图像的水平中心点

                # 遍历所有检测框，优先选择 "face" 或 "emblem"
                for box in r.boxes:
                    label = self.det_model.names[int(box.cls)]
                    if label not in ["face", "emblem"]:
                        continue  # 跳过非目标标签

                    # 计算当前 box 的中心点
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    box_center_x = (x1 + x2) // 2

                    # 根据标签判断方向
                    if label == "face" and box_center_x < image_center_x:
                        return True
                    elif label == "emblem" and box_center_x > image_center_x:
                        return True

        # 如果没有符合条件的 box，返回 False
        return False

    def process(self, img_path: str) -> Dict:
        """端到端处理流程"""
        # 输入验证
        if not os.path.exists(img_path):
            return {"status": "error", "message": "文件不存在"}

        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("图像读取失败")
        except Exception as e:
            return {"status": "error", "message": str(e)}

        # 执行分割
        seg_results = self.seg_model.predict(
            img_path,
            conf=0.5,
            retina_masks=True,
            imgsz=640,
            device=self.device
        )

        # 处理分割结果
        if not seg_results:
            return {"status": "error", "message": "未检测到身份证信息"}

        if seg_results[0].masks is None:
            return {"status": "error", "message": "分割掩模生成失败"}

        # 合并掩模
        combined_mask = self._merge_masks(seg_results[0], img.shape)

        # 图像矫正
        corrected_img = self._correct_perspective(img, combined_mask)
        if corrected_img is None:
            return {"status": "error", "message": "透视矫正失败"}

        # 方向校正
        if self._detect_orientation(corrected_img):
            corrected_img = cv2.rotate(corrected_img, cv2.ROTATE_180)

        # 保存结果
        save_path = self._save_result(corrected_img, img_path)
        return {"status": "success", "path": save_path}

    def _merge_masks(self, seg_result, img_shape: tuple) -> np.ndarray:
        """合并分割掩模（私有方法）"""
        h, w = img_shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        if seg_result.masks is None:
            return combined_mask  # 返回空掩模或抛出异常
        for mask in seg_result.masks:
            mask_data = mask.data[0].cpu().numpy()
            resized_mask = cv2.resize(mask_data, (w, h))
            _, binary_mask = cv2.threshold(resized_mask, 0.5, 255, cv2.THRESH_BINARY)
            combined_mask = cv2.bitwise_or(combined_mask, binary_mask.astype(np.uint8))
        return combined_mask

    def _save_result(self, img: np.ndarray, origin_path: str) -> str:
        """保存处理结果（私有方法）"""
        filename = os.path.basename(origin_path)
        name, ext = os.path.splitext(filename)
        save_path = os.path.join(self.save_dir, f"{name}_corrected{ext}")
        cv2.imwrite(save_path, img)
        return save_path

    def __del__(self):
        """资源清理"""
        if hasattr(self, 'seg_model'):
            del self.seg_model
        if hasattr(self, 'det_model'):
            del self.det_model
        torch.cuda.empty_cache()