import pickle
import cv2
import os
import threading
from keras_facenet import FaceNet
from mtcnn import MTCNN
import numpy as np
from scipy.spatial.distance import cosine
import time

class FaceRecognition:
    """
    初始化人脸识别类，无参数，已经集合进UI代码，该文件只作为测试使用！
    """
    def __init__(self):
        # 路径设置
        self.embeddings_dir = "FaceEmbeddings"
        self.known_faces_file = os.path.join(self.embeddings_dir, "known_face_embeddings.pkl")
        # 加载 MTCNN 检测器和 FaceNet 模型
        self.detector = MTCNN(
            min_face_size=100,  # 最小人脸尺寸
            steps_threshold=[0.7, 0.8, 0.9],  # 阈值设定
            scale_factor=0.7  # 缩放因子
        )
        self.facenet_model = FaceNet()
        # 加载已知人脸的特征和标签
        with open(self.known_faces_file, 'rb') as f:
            self.known_face_embeddings, self.known_face_labels = pickle.load(f)
        # 用于存储处理后的视频帧
        self.frame_to_process = None
        self.frame_lock = threading.Lock()
        # 用于检测是否需要退出
        self.exit_flag = False
        # 用于存储识别结果
        self.recognition_results = []

    def _face_recognition_thread(self):
        """人脸识别线程，处理视频帧中的人脸检测与分类。"""
        while not self.exit_flag:
            if self.frame_to_process is not None:
                # 复制帧进行处理，避免直接修改原始帧
                img_rgb = cv2.cvtColor(self.frame_to_process, cv2.COLOR_BGR2RGB)
                # 使用 MTCNN 检测人脸
                faces = self.detector.detect_faces(img_rgb)
                # 存储当前帧的人脸检测框和标签
                results = []
                for face in faces:
                    x, y, w, h = face['box']
                    face_img = img_rgb[y:y+h, x:x+w]
                    # 使用 Facenet 提取人脸特征
                    embedding = self.facenet_model.embeddings([face_img])[0]
                    # 计算与所有已知人脸的余弦相似度
                    distances = [cosine(embedding, known_embedding) for known_embedding in self.known_face_embeddings]
                    # 找到最小的距离（即最匹配的人脸）
                    min_distance_index = np.argmin(distances)
                    if distances[min_distance_index] < 0.5:  # 0.5 是阈值，低于这个值表示匹配
                        label = self.known_face_labels[min_distance_index]
                    else:
                        label = "Unknown"
                    results.append((x, y, w, h, label))
                # 锁定更新结果，防止与显示线程冲突
                with self.frame_lock:
                    self.recognition_results = results
            time.sleep(0.016)  # 控制线程每 30ms 处理一次（约 30 FPS）

    def display_video(self):
        """视频帧显示函数，持续显示视频流，只作为单独测试使用。"""
        cap = cv2.VideoCapture(0)
        while not self.exit_flag:
            ret, frame = cap.read()
            if not ret:
                break
            # 更新当前帧
            with self.frame_lock:
                self.frame_to_process = frame.copy()
            # 获取当前的识别结果
            with self.frame_lock:
                results = self.recognition_results
            # 在视频帧上绘制人脸框和标签
            for (x, y, w, h, label) in results:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            # 显示处理后的画面
            cv2.imshow('Face Recognition', frame)
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.exit_flag = True
                break
        cap.release()
        cv2.destroyAllWindows()

    def start_recognition(self):
        """启动人脸识别。启动线程进行识别和视频显示。"""
        # 启动人脸识别线程
        recognition_thread = threading.Thread(target=self._face_recognition_thread)
        recognition_thread.daemon = True  # 守护线程，在主程序结束时自动退出
        recognition_thread.start()
        # 启动视频显示
        self.display_video()

    def stop_recognition(self):
        """停止人脸识别，关闭摄像头显示。"""
        self.exit_flag = True

# 使用示例
if __name__ == "__main__":
    face_recog = FaceRecognition()
    # 启动人脸识别
    face_recog.start_recognition()
