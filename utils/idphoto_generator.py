import cv2
import numpy as np
from PIL import Image
import dlib
import mediapipe as mp
import os
from rembg import remove, new_session
import base64
import uuid
from io import BytesIO


class IdPhotoGenerator:
    def __init__(self, use_human_seg=True):
        """初始化证件照生成器"""
        self.use_human_seg = use_human_seg
        self.face_detection = None
        self.predictor = None
        self.session = None
        self._load_models()
        self.jpg_quality=95

    def _load_models(self):
        """加载所需模型"""
        try:
            # 初始化MediaPipe人脸检测
            mp_face_detection = mp.solutions.face_detection
            self.face_detection = mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5
            )

            # 加载dlib特征点检测器
            dlib_model_path = "../models/shape_predictor_68_face_landmarks.dat"
            if not os.path.exists(dlib_model_path):
                raise FileNotFoundError(f"dlib模型文件 {dlib_model_path} 未找到")
            self.predictor = dlib.shape_predictor(dlib_model_path)

            # 加载rembg会话
            model_name = "u2net_human_seg" if self.use_human_seg else "u2net"
            self.session = new_session(model_name)
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def _remove_background(self, image):
        """内部方法：去除背景"""
        try:
            if isinstance(image, np.ndarray):
                if image.shape[2] == 4:
                    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
                else:
                    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(image)

            result_img = remove(
                pil_img,
                session=self.session,
                post_process_mask=True,
                alpha_matting=True,
                alpha_matting_foreground_threshold=245,
                alpha_matting_background_threshold=15,
                alpha_matting_erode_size=15,
                alpha_matting_mask_size=30
            )
            return np.array(result_img)
        except Exception as e:
            raise RuntimeError(f"背景去除失败: {str(e)}")

    def _detect_face(self, image):
        """内部方法：人脸检测"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image[:, :, :3]
            results = self.face_detection.process(rgb_image)
            if not results.detections:
                raise ValueError("未检测到人脸")

            bbox = results.detections[0].location_data.relative_bounding_box
            h, w = image.shape[:2]
            padding = int(max(bbox.width, bbox.height) * w * 0.1)

            x1 = max(0, int(bbox.xmin * w) - padding)
            y1 = max(0, int(bbox.ymin * h) - padding)
            x2 = min(w, int((bbox.xmin + bbox.width) * w) + padding)
            y2 = min(h, int((bbox.ymin + bbox.height) * h) + padding)

            return (x1, y1, x2, y2)
        except Exception as e:
            raise RuntimeError(f"人脸检测失败: {str(e)}")

    def _align_face(self, image, face_box):
        """内部方法：人脸对齐"""
        try:
            x1, y1, x2, y2 = face_box
            face_region = image[y1:y2, x1:x2]

            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if face_region.shape[2] == 3 else face_region[:, :, 0]
            rect = dlib.rectangle(0, 0, face_region.shape[1], face_region.shape[0])
            landmarks = self.predictor(gray, rect)

            left_eye = (landmarks.part(36).x + landmarks.part(39).x) // 2, (landmarks.part(36).y + landmarks.part(39).y) // 2
            right_eye = (landmarks.part(42).x + landmarks.part(45).x) // 2, (landmarks.part(42).y + landmarks.part(45).y) // 2
            angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

            eyes_center = (x1 + (left_eye[0] + right_eye[0]) // 2,
                           y1 + (left_eye[1] + right_eye[1]) // 2)
            M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
            aligned = cv2.warpAffine(
                image, M, (image.shape[1], image.shape[0]),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
            return aligned
        except Exception as e:
            raise RuntimeError(f"人脸对齐失败: {str(e)}")

    def _resize_image(self, image, face_box):
        """内部方法：尺寸标准化"""
        try:
            x1, y1, x2, y2 = face_box
            face_height = y2 - y1
            target_size = (358, 441)
            target_face_height = int(target_size[1] * 0.5)
            scale = target_face_height / face_height

            new_w = int(image.shape[1] * scale)
            new_h = int(image.shape[0] * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            new_x1 = int(x1 * scale)
            new_y1 = int(y1 * scale)
            crop_x = max(0, new_x1 - (target_size[0] - (x2 - x1) * scale) // 2)
            crop_y = max(0, new_y1 - (target_size[1] - (y2 - y1) * scale) // 2)

            cropped = resized[int(crop_y):int(crop_y + target_size[1]),
                      int(crop_x):int(crop_x + target_size[0])]

            if cropped.shape[:2] != target_size:
                result = Image.new("RGBA", target_size, (255, 255, 255, 255))
                paste_x = (target_size[0] - cropped.shape[1]) // 2
                paste_y = (target_size[1] - cropped.shape[0]) // 2
                result.paste(Image.fromarray(cropped), (paste_x, paste_y))
                final_image = np.array(result)
            else:
                final_image = cropped

            return final_image
        except Exception as e:
            raise RuntimeError(f"尺寸调整失败: {str(e)}")

    def _prepare_jpg_image(self, image_array):
        """将RGBA图像转换为带白色背景的RGB JPG格式"""
        try:
            img = Image.fromarray(image_array)
            if img.mode == 'RGBA':
                # 创建白色背景
                background = Image.new('RGB', img.size, (255, 255, 255))
                # 合并图像
                background.paste(img, mask=img.split()[3])
                return background
            return img.convert('RGB')
        except Exception as e:
            raise RuntimeError(f"JPG格式转换失败: {str(e)}")

    def generate(self, input_data, is_base64=False):
        """
        生成证件照
        :param input_data: 图片路径或base64字符串
        :param is_base64: 是否使用base64输入
        :return: 包含状态码、信息和base64数据的字典
        """
        result = {"code": 200, "info": "success", "data": None}
        try:
            # 读取输入图像
            if is_base64:
                try:
                    # 去除base64头
                    if "," in input_data:
                        input_data = input_data.split(",")[1]
                    image_data = base64.b64decode(input_data)
                    np_arr = np.frombuffer(image_data, np.uint8)
                    image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                    if image is None:
                        raise ValueError("无效的base64图像数据")
                except Exception as e:
                    result.update({"code": 500, "info": f"base64解码失败: {str(e)}"})
                    return result
            else:
                image = cv2.imread(input_data, cv2.IMREAD_UNCHANGED)
                if image is None:
                    result.update({"code": 500, "info": f"无法读取图像文件: {input_data}"})
                    return result

            # 处理流程
            no_bg_image = self._remove_background(image)
            face_box = self._detect_face(no_bg_image)
            aligned_image = self._align_face(no_bg_image, face_box)
            final_face_box = self._detect_face(aligned_image)
            id_photo = self._resize_image(aligned_image, final_face_box)

            # 转换格式为JPG
            jpg_image = self._prepare_jpg_image(id_photo)

            # 保存到当前目录
            output_filename = f"id_photo_{uuid.uuid4().hex}.jpg"
            output_path = os.path.join(os.getcwd(), output_filename)
            jpg_image.save(output_path, "JPEG", quality=self.jpg_quality, optimize=True)

            # 生成base64
            buffered = BytesIO()
            jpg_image.save(buffered, format="JPEG", quality=self.jpg_quality)
            base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            result["data"] = base64_str
            result["info"] = f"证件照已生成并保存至: {output_path}"
        except Exception as e:
            result.update({"code": 500, "info": f"处理失败: {str(e)}"})

        return result


if __name__ == "__main__":
    # 使用示例
    generator = IdPhotoGenerator()

    # 使用文件路径
    result = generator.generate("test2.jpg")
    print(result)

    # # 使用base64
    # with open("input.jpg", "rb") as f:
    #     base64_data = base64.b64encode(f.read()).decode("utf-8")
    # result = generator.generate(base64_data, is_base64=True)
    # print(result)