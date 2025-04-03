import base64
from dataclasses import replace

import cv2
import re
from paddleocr import PaddleOCR

CHINESE_ETHNIC_GROUPS = {
    '汉', '蒙古', '回', '藏', '维吾尔', '苗', '彝', '壮', '布依', '朝鲜', '满', '侗', '瑶', '白', '土家',
    '哈尼', '哈萨克', '傣', '黎', '傈僳', '佤', '畲', '高山', '拉祜', '水', '东乡', '纳西', '景颇',
    '柯尔克孜', '土', '达斡尔', '仫佬', '羌', '布朗', '撒拉', '毛南', '仡佬', '锡伯', '阿昌', '普米',
    '塔吉克', '怒', '乌孜别克', '俄罗斯', '鄂温克', '德昂', '保安', '裕固', '京', '塔塔尔', '独龙', '鄂伦春',
    '赫哲', '门巴', '珞巴', '基诺'
}
PROVINCES = {
    '北京', '天津', '河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江',
    '上海', '江苏', '浙江', '安徽', '福建', '江西', '山东', '河南',
    '湖北', '湖南', '广东', '广西', '海南', '重庆', '四川', '贵州',
    '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆'
}


class OCRProcessor:
    def __init__(self, use_angle_cls=True, lang='ch'):
        """初始化OCR处理器"""
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)

    @staticmethod
    def _extract_info_from_ocr(ocr_data_list):
        """从OCR结果中提取结构化信息（私有方法）"""
        print("ocr_data_list", ocr_data_list)
        # 合并数组为字符串并预处理OCR错误
        merged_text = ((''.join(ocr_data_list).replace('民旅', '民族')
                       .replace('佳址', '住址')).replace('手用','出生')
                       .replace('性系','性别').replace('性址', '住址'))
        # 定义正则表达式模式（包含对OCR错误的容错）
        print(merged_text)
        patterns = {
            '姓名': r'姓名(.*?)(?=性别|民族|出生|住址|公民身份号码|$)',
            '性别': r'性别([男女])|性([男女])|性[^别]{0,3}([男女])',
            '民族': r'民[族旅]?([\u4e00-\u9fa5]{1,4})',
            '出生': r'(\d{4}年\d{1,2}月\d{1,2}日)',
            '住址': r'住址(.*?)(?=公民身份号码|$)',
            '公民身份号码': r'(\d{17}[\dXx])'
        }

        result = {}
        for field, pattern in patterns.items():
            match = re.search(pattern, merged_text, re.DOTALL)
            if match:
                # 提取值并去除前后空白
                value = match.group(1).strip()
                # 特殊处理住址可能包含换行符的情况
                if field == '住址':
                    value = re.sub(r'\s+', '', value)
                    # 自动补全省份（如果缺失）
                    if not any(value.startswith(p) for p in PROVINCES):
                        for p in PROVINCES:
                            if p in value:
                                value = value[value.index(p):]
                                break
                if field == '民族':
                    value = match.group(1).strip()
                    # 检查是否是有效的民族
                    for valid_ethnic in CHINESE_ETHNIC_GROUPS:
                        if valid_ethnic.startswith(value) or value.startswith(valid_ethnic):
                            value = valid_ethnic
                            break
                result[field] = value

            else:
                result[field] = '未提取到'

        return result

    def process_id_card(self, image_path):
        """处理身份证识别的主方法"""
        # 读取图像并初始化结果
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像文件: {image_path}")

        # 初次OCR识别
        primary_result = self.ocr.ocr(image_path, cls=True)
        if primary_result[0] is None:
            return {
                'success': False,
                'info':''
            }
        texts = [item[1][0] for item in primary_result[0]] if primary_result else []
        # 方向判断逻辑
        reverse = False
        if len(texts) > 1:
            # 检查前两个文本块是否包含关键字段
            if any('公民身份号码' in texts[i] for i in range(min(2, len(texts)))):
                reverse = True
                texts = texts[::-1]

        # 信息提取
        info = self._extract_info_from_ocr(texts)

        # 图像旋转处理
        final_image = image
        if reverse:
            # 旋转图像并重新识别
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
            rotated_result = self.ocr.ocr(rotated_image, cls=True)
            rotated_texts = [item[1][0] for item in rotated_result[0]] if rotated_result else []
            info = self._extract_info_from_ocr(rotated_texts)
            final_image = rotated_image

        # 转换为base64
        _, buffer = cv2.imencode('.jpg', final_image)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        return {
            'success': True,
            'info': info,
            'image_base64': base64_image,
        }