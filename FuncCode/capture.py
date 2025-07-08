import cv2, os

class FaceCapture:
    """
    人脸采集类，用于通过摄像头捕获人脸图像并保存，只作为测试时使用，UI以及最终代码中不包含此代码！
    
    参数:
        save_dir (str): 保存图像的目录路径，默认。
        person_name (str): 被采集者的姓名，用于生成文件名，需要修改。
        target_count (int): 需要采集的图像数量，默认为5张，可修改。
    """
    def __init__(self, save_dir="KnownFaces", person_name="unknown", target_count=5):
        self.save_dir = os.path.join(save_dir, person_name)
        self.person_name = person_name
        self.target_count = target_count
        self.cap = None
        self.count = 0
        self.window_title = f"Press SPACE to Capture ({self.count}/{self.target_count})"
        os.makedirs(self.save_dir, exist_ok=True)
    
    def initialize_camera(self, camera_index=0):
        """初始化摄像头"""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("错误：无法打开摄像头")
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
    
    def capture_faces(self):
        """执行人脸采集过程"""
        if self.cap is None:
            self.initialize_camera()
        try:
            while self.count < self.target_count:
                ret, frame = self.cap.read()
                if not ret:
                    print("无法获取画面")
                    break
                # 显示实时画面
                cv2.imshow(self.window_title, frame)
                # 检测按键
                key = cv2.waitKey(30)
                if key == 32:  # 空格键
                    self._capture_frame(frame)
                elif key == 27:  # ESC键
                    print("用户中断拍摄")
                    break
            print(f"拍摄完成，共保存 {self.count} 张照片到 {self.save_dir}")
        finally:
            self._release_resources()
    
    def _capture_frame(self, frame):
        """捕获并保存当前帧"""
        self.count += 1
        save_path = os.path.join(self.save_dir, f"{self.person_name}{self.count}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"第 {self.count}/{self.target_count} 张图片已保存至: {save_path}")
        # 更新窗口标题
        self._update_window_title()
        self._show_captured_photo(frame)
    
    def _update_window_title(self):
        """更新窗口标题显示进度"""
        self.window_title = f"Press SPACE to Capture ({self.count}/{self.target_count})"
        cv2.setWindowTitle(self.window_title, self.window_title)
    
    def _show_captured_photo(self, frame):
        """短暂显示捕获的照片"""
        temp_window = f"Captured Photo {self.count}"
        cv2.imshow(temp_window, frame)
        cv2.waitKey(500)  # 显示0.5秒
        cv2.destroyWindow(temp_window)
    
    def _release_resources(self):
        """释放摄像头资源和关闭窗口"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def __del__(self):
        """析构函数，确保资源被释放"""
        self._release_resources()


# 测试代码
if __name__ == "__main__":
    try:
        # 创建采集实例
        face_capture = FaceCapture(
            save_dir="KnownFaces",
            person_name="szh",
            target_count=5)
        # 开始采集
        face_capture.capture_faces()
    except Exception as e:
        print(f"发生错误: {e}")