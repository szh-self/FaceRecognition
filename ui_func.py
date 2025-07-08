from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDateTime
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QHeaderView, QTableWidgetItem, QTableWidget, QMessageBox
from PyQt5.QtChart import QChart, QBarSeries, QBarSet, QBarCategoryAxis, QValueAxis
from GUIFile.ui_format import Ui_Form
from FuncCode.train import FaceEmbeddingExtractor
from FuncCode.database import RecognitionDatabase
from keras_facenet import FaceNet
from mtcnn import MTCNN
from scipy.spatial.distance import cosine
import numpy as np
import cv2, os, pickle, threading, time, shutil, sys

# 确保必要的目录存在
os.makedirs("FaceEmbeddings", exist_ok=True)    # 存储人脸特征向量的目录
os.makedirs("KnownFaces", exist_ok=True)        # 存储已知人脸模板的目录
# db = RecognitionDatabase()                    # 为了线程安全放到mainwindow中

class VideoThread(QThread):
    """视频捕获线程，负责从摄像头读取帧并进行处理"""
    change_pixmap_signal = pyqtSignal(np.ndarray)       # 信号：通知Qt主线程
    def __init__(self, recognition_system=None, capture_mode=False):
        super().__init__()
        self.recognition_system = recognition_system    # 人脸识别系统实例
        self.capture_mode = capture_mode                # 是否为捕获模式（仅显示不识别）
        self._run_flag = True                           # 线程运行标志
        
    def run(self):
        """主循环：从摄像头捕获帧并处理"""
        cap = cv2.VideoCapture(0)  # 打开默认摄像头
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break
            if self.capture_mode:
                # 捕获模式：直接发送帧用于显示
                self.change_pixmap_signal.emit(frame)
                time.sleep(0.03)
                continue
            if self.recognition_system:
                # 识别模式：处理人脸识别
                with self.recognition_system.frame_lock:
                    self.recognition_system.frame_to_process = frame.copy()
                    results = self.recognition_system.recognition_results
                # 在帧上绘制识别结果（矩形框和标签）
                for (x, y, w, h, label) in results:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                    self.recognition_system.window.db.add_record(label)
            self.change_pixmap_signal.emit(frame)
            time.sleep(0.03)  # 控制帧率，30fps
        cap.release()
        
    def stop(self):
        """停止线程"""
        self._run_flag = False
        self.wait()

class FaceRecognition:
    """人脸识别系统核心类"""
    def __init__(self, window):
        self.window = window
        self.embeddings_dir = "FaceEmbeddings"
        self.known_faces_file = os.path.join(self.embeddings_dir, "known_face_embeddings.pkl")
        # 初始化模型
        self.detector = MTCNN(min_face_size=100, steps_threshold=[0.7, 0.8, 0.9], scale_factor=0.8)
        self.facenet_model = FaceNet()
        # 存储已知人脸数据
        self.known_face_embeddings = []  # 人脸特征向量列表
        self.known_face_labels = []      # 对应的人脸标签列表
        # 线程相关变量
        self.frame_to_process = None        # 待处理的视频帧
        self.frame_lock = threading.Lock()  # 线程锁
        self.exit_flag = False              # 退出标志
        self.recognition_results = []       # 识别结果存储
        self.video_thread = None            # 视频线程
        self.recognition_thread = None      # 识别线程
        # 加载已知人脸数据
        self.load_known_faces()         

    def load_known_faces(self):
        """从文件加载已知人脸特征向量和标签"""
        if os.path.exists(self.known_faces_file):
            try:
                with open(self.known_faces_file, 'rb') as f:
                    self.known_face_embeddings, self.known_face_labels = pickle.load(f)
            except Exception as e:
                print(f"Error loading known faces: {e}")
                self.known_face_embeddings = []
                self.known_face_labels = []

    def save_known_faces(self):
        """保存已知人脸特征向量和标签到文件"""
        os.makedirs(self.embeddings_dir, exist_ok=True)
        with open(self.known_faces_file, 'wb') as f:
            pickle.dump((self.known_face_embeddings, self.known_face_labels), f)

    def add_face_embedding(self, image_path, label):
        """添加新人脸特征向量到已知列表"""
        img = cv2.imread(image_path)
        if img is None:
            return False
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(img_rgb)
        if not faces:
            return False
        face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
        x, y, w, h = face['box']
        face_img = img_rgb[y:y+h, x:x+w]
        # 提取特征向量
        embedding = self.facenet_model.embeddings([face_img])[0]
        self.known_face_embeddings.append(embedding)
        self.known_face_labels.append(label)
        self.save_known_faces()
        return True

    def train(self):
        """训练模型，并重新加载已知人脸"""
        self.load_known_faces()
        return len(self.known_face_labels) > 0

    def _face_recognition_thread(self):
        """人脸识别线程的主循环"""
        while not self.exit_flag:
            if self.frame_to_process is not None:
                img_rgb = cv2.cvtColor(self.frame_to_process, cv2.COLOR_BGR2RGB)
                faces = self.detector.detect_faces(img_rgb)
                results = []
                for face in faces:
                    x, y, w, h = face['box']
                    face_img = img_rgb[y:y+h, x:x+w]
                    embedding = self.facenet_model.embeddings([face_img])[0]
                    if self.known_face_embeddings:
                        # 计算与已知人脸的余弦距离
                        distances = [cosine(embedding, known_embedding) 
                                   for known_embedding in self.known_face_embeddings]
                        min_distance_index = np.argmin(distances)
                        # 根据阈值判断是否匹配，较高精度阈值0.3
                        label = self.known_face_labels[min_distance_index] if distances[min_distance_index] < 0.3 else "Unknown"
                    else:
                        label = "Unknown"
                    results.append((x, y, w, h, label))
                with self.frame_lock:
                    self.recognition_results = results
            time.sleep(0.03)

    def start_recognition(self):
        """启动人脸识别系统"""
        self.exit_flag = False
        self.recognition_thread = threading.Thread(target=self._face_recognition_thread)
        self.recognition_thread.daemon = True
        self.recognition_thread.start()
        self.video_thread = VideoThread(self)
        self.video_thread.change_pixmap_signal.connect(self.window.update_image)
        self.video_thread.start()

    def stop_recognition(self):
        """停止人脸识别系统"""
        self.exit_flag = True
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        if self.recognition_thread and self.recognition_thread.is_alive():
            self.recognition_thread.join(timeout=1.0)
            self.recognition_thread = None

class MainWindow(QtWidgets.QMainWindow):
    """主窗口类，处理用户界面交互"""
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle("人脸识别系统")
        
        # 连接信号与槽
        self.ui.btn_add.clicked.connect(self.add_face_template)
        self.ui.btn_capture.clicked.connect(self.capture_face)
        self.ui.btn_save.clicked.connect(self.save_face_templates)
        self.ui.btn_train.clicked.connect(self.train_model)
        self.ui.btn_delete.clicked.connect(self.delete_template)
        self.ui.btn_start.clicked.connect(self.start_recognition)
        self.ui.btn_end.clicked.connect(self.end_recognition)
        self.ui.refresh_btn.clicked.connect(self.update_chart)
        
        # 初始化按钮状态
        self.ui.btn_capture.setEnabled(False)
        self.ui.btn_save.setEnabled(False)
        self.ui.btn_train.setEnabled(False)
        self.ui.btn_end.setEnabled(False)
        self.ui.lineEdit_name.setEnabled(False)

        # 初始化数据库类
        self.db = RecognitionDatabase()

        # 初始化人脸识别系统
        self.recognize = FaceRecognition(window=self)
        self.capture_thread = None
        self.captured_image = None  # 捕获的原始图像（BGR格式）
        self.captured_face = None   # 捕获的人脸区域（BGR格式）
        self.last_frame = None      # 最后一帧视频（用于捕获）

        # 初始化模型训练系统
        self.train = FaceEmbeddingExtractor()

        # 初始化表格
        self.init_table()
        self.ui.tabWidget.setCurrentIndex(0)
        
        # 初始加载图表
        self.update_chart()
    
    def init_table(self):
        """初始化模板表格"""
        self.ui.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.ui.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.tableWidget.setEditTriggers(QTableWidget.NoEditTriggers)
        # 加载已有模板
        folders = self.get_tpl_name("KnownFaces")
        for folder in folders:
            row_position = self.ui.tableWidget.rowCount()
            self.ui.tableWidget.insertRow(row_position)
            self.write_to_item_cells(row_position, folder)
        # 检查是否有可训练的模板
        if self.ui.tableWidget.rowCount() > 0:
            self.ui.btn_train.setEnabled(True)
        else:
            self.ui.btn_train.setEnabled(False)
            self.ui.btn_start.setEnabled(False)
            self.ui.btn_delete.setEnabled(False)
    
    def get_tpl_name(self, path):
        """获取指定路径下的所有文件夹名称"""
        with os.scandir(path) as entries:
            return [entry.name for entry in entries if entry.is_dir()]
    
    def write_to_item_cells(self, write_row, write_item):
        """向表格写入数据"""
        self.ui.tableWidget.setItem(write_row, 0, QTableWidgetItem(write_item))
        # 设置文本居中，并无法修改
        for row in range(self.ui.tableWidget.rowCount()):
            item = self.ui.tableWidget.item(row, 0)
            if item:
                item.setTextAlignment(Qt.AlignCenter)
        
    def update_image(self, cv_img):
        """更新显示的图像"""
        self.last_frame = cv_img.copy()
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qt_img = self.convert_cv_qt(rgb_image)
        self.ui.label_video.setPixmap(qt_img)
        
    def convert_cv_qt(self, cv_img):
        """将OpenCV图像转换为QPixmap"""
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(cv_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.ui.label_video.size(), Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(p)
        
    def add_face_template(self):
        """进入添加人脸模板模式"""
        self.recognize.stop_recognition()
        self.ui.btn_capture.setEnabled(True)
        self.ui.btn_save.setEnabled(False)
        self.ui.lineEdit_name.setEnabled(True)
        self.ui.btn_add.setEnabled(False)
        self.ui.btn_start.setEnabled(False)
        self.ui.btn_end.setEnabled(False)
        self.ui.btn_train.setEnabled(False)
        self.ui.btn_delete.setEnabled(False)
        self.ui.label_video.setText("摄像头准备中...")
        self.start_capture()
        
    def start_capture(self):
        """启动视频捕获线程"""
        if self.capture_thread:
            self.capture_thread.stop()
        self.capture_thread = VideoThread(capture_mode=True)
        self.capture_thread.change_pixmap_signal.connect(self.update_image)
        self.capture_thread.start()
        
    def capture_face(self):
        """捕获当前帧中的人脸"""
        if self.last_frame is None:
            QMessageBox.warning(self, "警告", "没有可捕获的图像")
            return
        # 停止视频捕获线程
        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread = None
        # 使用原始帧（BGR格式）
        frame_bgr = self.last_frame.copy()
        # 转换为RGB用于人脸检测
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = self.recognize.detector.detect_faces(img_rgb)
        if not faces:
            QMessageBox.warning(self, "警告", "未检测到人脸，请重新拍摄")
            self.start_capture()
            return
        # 选择最大的脸
        face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)  # 确保坐标不越界
        w, h = min(w, img_rgb.shape[1] - x), min(h, img_rgb.shape[0] - y)
        # 保存原始图像和人脸区域
        self.captured_image = frame_bgr.copy()
        self.captured_face = frame_bgr[y:y+h, x:x+w].copy()
        # 在图像上绘制标记
        cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 绿色框
        cv2.putText(frame_bgr, "Captured", (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)  # 黄色文字
        # 显示处理后的图像
        display_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qt_img = self.convert_cv_qt(display_img)
        self.ui.label_video.setPixmap(qt_img)
        self.ui.btn_save.setEnabled(True)

    def save_face_templates(self):
        """保存人脸模板"""
        name = self.ui.lineEdit_name.text().strip()
        if not name:
            QMessageBox.warning(self, "警告", "请输入姓名")
            return
        template_dir = os.path.join("KnownFaces", name)
        # 检查是否已存在同名模板
        if os.path.exists(template_dir):
            reply = QMessageBox.question(self, "确认", 
                                        f"已存在名为 '{name}' 的模板，是否覆盖?",
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
            shutil.rmtree(template_dir)
        os.makedirs(template_dir, exist_ok=True)
        # 保存原始图像
        if self.captured_image is not None:
            original_path = os.path.join(template_dir, f"{name}.jpg")
            cv2.imwrite(original_path, self.captured_image)
            self.recognize.add_face_embedding(original_path, name)
        # 更新表格
        self.update_table(name)
        self.reset_add_face_state()
        self.ui.btn_train.setEnabled(True)
        QMessageBox.information(self, "成功", f"人脸模板 '{name}' 已成功保存")
        
    def train_model(self):
        """训练人脸识别模型"""
        self.ui.label_video.setText("正在训练模型...")
        QtWidgets.QApplication.processEvents()  # 强制更新UI
        # 在新线程中执行训练
        train_thread = threading.Thread(target=self._train_in_thread)
        train_thread.daemon = True
        train_thread.start()
        
    def _train_in_thread(self):
        """在后台线程中执行训练"""
        success = self.train.extract_embeddings()
        self.ui.label_video.setText("训练完成！可以开始识别！")
        self.ui.btn_train.setEnabled(False)
        self.ui.btn_start.setEnabled(True)
        self.ui.btn_delete.setEnabled(True)

    def delete_template(self):
        """删除选定的人脸模板"""
        selected_items = self.ui.tableWidget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择要删除的模板")
            return
        name = selected_items[0].text()
        reply = QMessageBox.question(self, "确认删除", 
                                    f"确定要删除模板 '{name}' 吗?\n这将删除所有相关文件。",
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No:
            return
        try:
            # 删除模板目录
            template_dir = os.path.join("KnownFaces", name)
            if os.path.exists(template_dir):
                shutil.rmtree(template_dir)
            # 从特征向量文件中移除该模板
            if os.path.exists(self.recognize.known_faces_file):
                with open(self.recognize.known_faces_file, 'rb') as f:
                    embeddings, labels = pickle.load(f)
                # 过滤掉要删除的模板
                new_embeddings = []
                new_labels = []
                for emb, lbl in zip(embeddings, labels):
                    if lbl != name:
                        new_embeddings.append(emb)
                        new_labels.append(lbl)
                # 保存更新后的特征向量
                with open(self.recognize.known_faces_file, 'wb') as f:
                    pickle.dump((new_embeddings, new_labels), f)
                # 更新内存中的数据
                self.recognize.known_face_embeddings = new_embeddings
                self.recognize.known_face_labels = new_labels
            # 从表格中移除
            for row in range(self.ui.tableWidget.rowCount()):
                if self.ui.tableWidget.item(row, 0).text() == name:
                    self.ui.tableWidget.removeRow(row)
                    break
            # 检查是否还有模板可以训练
            self.ui.btn_train.setEnabled(self.ui.tableWidget.rowCount() > 0)
            self.ui.btn_delete.setEnabled(self.ui.tableWidget.rowCount() > 0)
            self.ui.btn_start.setEnabled(self.ui.tableWidget.rowCount() > 0)
            QMessageBox.information(self, "成功", f"模板 '{name}' 已删除")
            self.db.delete_records_by_name(name)
            self.update_chart()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"删除模板时出错: {str(e)}")

    def update_table(self, name):
        """更新模板表格"""
        existing = False
        # 检查是否已存在同名模板
        for row in range(self.ui.tableWidget.rowCount()):
            if self.ui.tableWidget.item(row, 0).text() == name:
                existing = True
                break
        if not existing:
            row_position = self.ui.tableWidget.rowCount()
            self.ui.tableWidget.insertRow(row_position)
            self.write_to_item_cells(row_position, name)
        
    def reset_add_face_state(self):
        """重置添加人脸模板的状态"""
        self.ui.lineEdit_name.clear()
        self.ui.btn_capture.setEnabled(False)
        self.ui.btn_save.setEnabled(False)
        self.ui.btn_end.setEnabled(False)
        self.ui.lineEdit_name.setEnabled(False)
        self.ui.btn_add.setEnabled(True)
        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread = None
        self.captured_image = None
        self.captured_face = None
        
    def start_recognition(self):
        """启动人脸识别"""
        self.reset_add_face_state()
        self.ui.btn_start.setEnabled(False)
        self.ui.btn_end.setEnabled(True)
        self.ui.btn_add.setEnabled(False)
        self.ui.btn_train.setEnabled(False)
        self.ui.btn_delete.setEnabled(False)
        self.recognize.start_recognition()
        
    def end_recognition(self):
        """停止人脸识别"""
        self.ui.btn_start.setEnabled(True)
        self.ui.btn_end.setEnabled(False)
        self.ui.btn_add.setEnabled(True)
        self.ui.btn_delete.setEnabled(True)
        self.recognize.stop_recognition()
        # 强制立即更新图表
        self.update_chart()
    
    def update_chart(self):
        """更新图表数据：根据选择的时间段显示不同粒度的数据"""
        # 获取当前选中的时间段
        if self.ui.btn_hour.isChecked():
            time_range = "hour"
        elif self.ui.btn_day.isChecked():
            time_range = "day"
        elif self.ui.btn_week.isChecked():
            time_range = "week"
        elif self.ui.btn_month.isChecked():
            time_range = "month"
        elif self.ui.btn_year.isChecked():
            time_range = "year"
        else:
            time_range = "day"  # 默认
        # 计算开始和结束时间
        end_time = QDateTime.currentDateTime()
        if time_range == "hour":
            start_time = end_time.addSecs(-3600)  # 1小时前
            time_format = "yyyy-MM-dd HH:mm"
            time_group = lambda r: r['time'][11:16]  # 提取小时和分钟
            # 生成所有分钟点（确保是字符串列表）
            all_time_points = [
                str(end_time.addSecs(-60 * i).toString("HH:mm"))
                for i in range(60, -1, -1)
            ]
        elif time_range == "day":
            now = QDateTime.currentDateTime()
            # 设置 end_time 为当前小时整点
            end_time = QDateTime(now.date(), QtCore.QTime(now.time().hour()+1, 0))
            # 设置 start_time 为 24 小时前的整点
            start_time = end_time.addSecs(-3600 * 24)
            time_format = "yyyy-MM-dd HH:00"
            time_group = lambda r: r['time'][11:13] + ":00"  # 提取小时
            # 生成从 start_time 到 end_time 的 25 个整点小时段（含起止点）
            all_time_points = [
                str(start_time.addSecs(3600 * i).toString("HH:00"))
                for i in range(25)
            ]
        elif time_range == "week":
            start_time = end_time.addDays(-7)
            time_format = "yyyy-MM-dd"
            time_group = lambda r: r['time'][5:10]  # 提取月-日
            # 生成所有日期点（确保是字符串列表）
            all_time_points = [
                str(end_time.addDays(-i).toString("MM-dd"))
                for i in range(7, -1, -1)
            ]
        elif time_range == "month":
            start_time = end_time.addMonths(-1)
            time_format = "yyyy-MM-dd"
            time_group = lambda r: r['time'][5:10]  # 提取月-日
            # 生成所有日期点（确保是字符串列表）
            all_time_points = [
                str(start_time.addDays(i).toString("MM-dd"))
                for i in range(31) if start_time.addDays(i) <= end_time
            ]
        else:  # year
            # 从当前月份开始计算过去12个月
            current_month = end_time.date().month()
            current_year = end_time.date().year()
            # 设置结束时间为当前月份的最后一天
            end_date = QtCore.QDate(current_year, current_month, 1).addMonths(1).addDays(-1)
            end_time = QDateTime(end_date)
            # 设置开始时间为12个月前
            start_date = QtCore.QDate(current_year, current_month, 1).addMonths(-11)
            start_time = QDateTime(start_date)
            time_format = "yyyy-MM"
            time_group = lambda r: r['time'][:7]  # 提取年-月
            # 生成所有月份点（确保是字符串列表）
            all_time_points = [
                str(start_date.addMonths(i).toString("yyyy-MM"))
                for i in range(12)
            ]
        # 从数据库获取数据
        records = self.db.get_records_by_time_range(
            start_time.toString("yyyy-MM-dd HH:mm:ss"),
            end_time.toString("yyyy-MM-dd HH:mm:ss")
        )
        # 收集所有name
        names = set(record['name'] for record in records) if records else set()
        # 初始化数据结构，确保所有时间点都有记录
        name_time_counts = {name: {tp: 0 for tp in all_time_points} for name in names}
        # 填充实际数据
        for record in records:
            time_point = time_group(record)
            name = record['name']
            # 确保时间点在预定义的范围内
            if time_point in name_time_counts.get(name, {}):
                name_time_counts[name][time_point] += 1
        # 创建柱状图系列
        chart = QChart()
        chart.setTitle(f"人脸识别统计 ({start_time.toString(time_format)} 至 {end_time.toString(time_format)})")
        chart.setAnimationOptions(QChart.SeriesAnimations)
        # 为每个name创建一个QBarSet
        bar_sets = {}
        names_sorted = sorted(names) if names else []
        # 动态生成颜色方案
        def generate_colors(num_colors):
            colors = []
            for i in range(num_colors):
                hue = i * (360 / max(1, num_colors))
                saturation = 80 + (i % 3) * 10
                value = 100 - (i % 2) * 10
                color = QtGui.QColor.fromHsv(int(hue) % 360, int(saturation), int(value))
                colors.append(color)
            return colors
        color_list = generate_colors(len(names_sorted))
        for i, name in enumerate(names_sorted):
            bar_set = QBarSet(name)
            bar_set.setColor(color_list[i % len(color_list)])
            # 按预定义的时间顺序填充数据
            counts = [name_time_counts[name][tp] for tp in all_time_points]
            bar_set.append(counts)
            bar_sets[name] = bar_set
        # 创建QBarSeries并添加所有QBarSet
        series = QBarSeries()
        for name in names_sorted:
            series.append(bar_sets[name])
        # 设置柱子上方显示数值
        series.setLabelsVisible(True)
        series.setLabelsFormat("@value")
        series.setLabelsPosition(QBarSeries.LabelsCenter)
        chart.addSeries(series)
        # 创建坐标轴
        axis_x = QBarCategoryAxis()
        # 确保all_time_points是字符串列表
        axis_x.append([str(tp) for tp in all_time_points])  # 显式转换为字符串列表
        axis_x.setLabelsAngle(90)  # 90度垂直显示
        if time_range == "hour":
            axis_x.setLabelsFont(QtGui.QFont("Arial", 4))  # 设置小一点的字体
        else:
            axis_x.setLabelsFont(QtGui.QFont("Arial", 8))  # 设置大一点的字体
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)
        # 调整图表边距，为垂直标签留出空间
        # chart.setMargins(QtCore.QMargins(10, 10, 10, 40))  # 下边距增大
        axis_y = QValueAxis()
        axis_y.setLabelFormat("%d")
        axis_y.setTitleText("识别次数")
        # 计算最大y值
        max_value = max(
            (count for name_counts in name_time_counts.values() for count in name_counts.values()),
            default=0
        )
        axis_y.setRange(0, max_value + 1)
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)
        # 添加图例
        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignBottom)
        # 设置图表视图
        self.ui.chart_view.setChart(chart)
        self.ui.chart_view.update()

    def closeEvent(self, event):
        """窗口关闭事件处理"""
        self.recognize.stop_recognition()
        if self.capture_thread:
            self.capture_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())