import os
import numpy as np
from keras.models import load_model
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
import pickle
import cv2

class FaceEmbeddingExtractor:
    """初始化人脸特征提取器，UI中会调用此代码。"""
    def __init__(self):
        self.known_faces_dir = "KnownFaces"
        self.embeddings_dir = "FaceEmbeddings"

        # 加载 MTCNN 面部检测器和 FaceNet 模型
        self.detector = MTCNN()
        self.facenet_model = FaceNet()

        # 用于存储所有已知人脸的特征和标签
        self.known_face_embeddings = []
        self.known_face_labels = []

    def extract_embeddings(self):
        """提取所有已知人脸的特征并保存。"""
        # 遍历 KnownFaces 文件夹，提取每个文件夹中的人脸特征
        self.known_face_embeddings = []
        self.known_face_labels = []
        for person_name in os.listdir(self.known_faces_dir):
            person_folder = os.path.join(self.known_faces_dir, person_name)
            if os.path.isdir(person_folder):  # 只处理文件夹
                for img_name in os.listdir(person_folder):
                    img_path = os.path.join(person_folder, img_name)
                    # 读取图片
                    img = cv2.imread(img_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # 使用 MTCNN 检测人脸
                    faces = self.detector.detect_faces(img_rgb)
                    if faces:  # 如果检测到人脸
                        for face in faces:
                            x, y, w, h = face['box']
                            face_img = img_rgb[y:y+h, x:x+w]
                            # 使用 Facenet 提取人脸特征
                            embedding = self.facenet_model.embeddings([face_img])[0]
                            # 存储特征和标签
                            self.known_face_embeddings.append(embedding)
                            self.known_face_labels.append(person_name)
        # 保存提取的特征到文件
        if not os.path.exists(self.embeddings_dir):
            os.makedirs(self.embeddings_dir)
        embedding_file = os.path.join(self.embeddings_dir, "known_face_embeddings.pkl")
        with open(embedding_file, 'wb') as f:
            pickle.dump((self.known_face_embeddings, self.known_face_labels), f)
        print(f"已知人脸特征已保存到 {embedding_file}")
        print(f"已知人脸标签：{self.known_face_labels}")
        print(f"已知人脸特征数：{len(self.known_face_embeddings)}")
        return embedding_file

    def get_known_face_embeddings(self):
        """
        获取已知人脸的特征和标签，作为测试验证用。
        
        :return: (known_face_embeddings, known_face_labels)
        """
        return self.known_face_embeddings, self.known_face_labels
    

# 测试代码
if __name__ == "__main__":
    # 创建提取器实例
    face_extractor = FaceEmbeddingExtractor()
    # 提取并保存已知人脸特征
    embedding_file = face_extractor.extract_embeddings()
    print(f"特征已保存到：{embedding_file}")

    
