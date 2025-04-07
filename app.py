import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from utils.ocr_processor import OCRProcessor
from utils.idcard_processor import IDCardProcessor
from utils.idphoto_generator import IdPhotoGenerator
import cv2
import base64
import torch

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB限制

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 初始化处理器（应用启动时只初始化一次）
id_processor = IDCardProcessor(
    save_dir=UPLOAD_FOLDER,
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)

ocr_processor = OCRProcessor(use_angle_cls=True, lang='ch')
photo_generator = IdPhotoGenerator()
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 主框架路由
@app.route('/')
def index():
    return render_template('layout.html')

# 身份证识别页面路由
@app.route('/id-card')
def id_card_page():
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    return render_template('id_card.html', ajax_request=is_ajax)
@app.route('/id-card/process', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        # 检查是否选择了文件
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            # 保存原始文件
            filename = secure_filename(file.filename)
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(original_path)

            try:
                # 1. 先进行身份证裁剪和矫正
                result = id_processor.process(original_path)
                if result['status'] != 'success':
                    return jsonify({'error': result['message']}), 500

                corrected_path = result['path']
                # 2. 进行OCR识别
                ocr_result = ocr_processor.process_id_card(corrected_path)
                if not ocr_result['success']:
                    # 矫正后的图像
                    corrected_img = cv2.imread(corrected_path)
                    _, corrected_buffer = cv2.imencode('.jpg', corrected_img)
                    corrected_base64 = base64.b64encode(corrected_buffer).decode('utf-8')
                    ocr_result['image_base64'] = corrected_base64
                # 3. 准备返回数据
                # 原始图像转为base64
                original_img = cv2.imread(original_path)
                _, original_buffer = cv2.imencode('.jpg', original_img)
                original_base64 = base64.b64encode(original_buffer).decode('utf-8')
                response_data = {
                    'original_image': original_base64,
                    'corrected_image': ocr_result['image_base64'],
                    'ocr_result': ocr_result['info'],
                    'original_filename': filename,
                    'corrected_filename': os.path.basename(corrected_path)
                }

                return jsonify(response_data)

            except Exception as e:
                return jsonify({'error': str(e)}), 500
    return render_template('layout.html')


# 其他功能页面的路由可以在这里添加
@app.route('/photo-gen')
def photo_gen_page():
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    return render_template('photo-gen.html', ajax_request=is_ajax)

@app.route('/id-photo/process', methods=['GET', 'POST'])
def photo_gen_process():
    if request.method == 'POST':
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        # 检查是否选择了文件
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            try:
                file_data = file.read()
                base64_data = base64.b64encode(file_data).decode('utf-8')
                # 生成证件照
                result = photo_generator.generate(base64_data, is_base64=True)
                return jsonify(result)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    return render_template('layout.html')

@app.route('/face-compare')
def face_compare_page():
    return "<h1>人脸比对功能开发中</h1>"

@app.route('/liveness')
def liveness_page():
    return "<h1>活体检测功能开发中</h1>"

@app.route('/face-db')
def face_db_page():
    return "<h1>人脸数据库功能开发中</h1>"

if __name__ == '__main__':
    app.run(debug=True)