import sqlite3
from datetime import datetime
import threading, os, time
from queue import Queue
from contextlib import contextmanager

class RecognitionDatabase:
    """
    线程安全的SQLite数据库连接池
    功能：提供人脸识别记录的存储和查询功能，支持多线程安全访问
    
    参数：
        db_name: 数据库文件路径
    """
    _local = threading.local()  # 线程本地存储
    
    def __init__(self, db_name='DataBase/recognition_records.db'):
        """初始化数据库连接池"""
        self.db_name = db_name
        os.makedirs(os.path.dirname(db_name), exist_ok=True)  # 确保目录存在
        self._connection_pool = Queue(maxsize=5)  # 连接池大小设置为5
        self._initialize_pool()  # 初始化连接池
        self._create_table()  # 创建数据表
    
    def _initialize_pool(self):
        """初始化连接池，创建5个数据库连接"""
        for _ in range(5):
            conn = sqlite3.connect(
                self.db_name,
                check_same_thread=False,  # 允许不同线程使用同一连接
                isolation_level=None,     # 自动提交模式
                timeout=10.0              # 连接超时时间
            )
            conn.execute("PRAGMA journal_mode=WAL")  # 启用WAL模式提高并发性能
            self._connection_pool.put(conn)  # 将连接放入池中
    
    @contextmanager
    def _get_connection(self):
        """
        获取数据库连接（上下文管理器）
        确保连接使用后自动归还到连接池
        """
        conn = self._connection_pool.get()  # 从池中获取连接
        try:
            yield conn  # 返回连接给调用者使用
        finally:
            self._connection_pool.put(conn)  # 使用完毕后归还连接
    
    def _create_table(self):
        """创建人脸识别记录表（无id字段）"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 创建表，直接使用name和time作为复合主键
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS recognition_records 
            (
                name TEXT NOT NULL,
                time TEXT NOT NULL,
                PRIMARY KEY (name, time)  -- 复合主键替代id
            ) WITHOUT ROWID  -- 优化存储结构
            """)
            conn.commit()
    
    def add_record(self, name):
        """
        添加一条识别记录
        
        参数：
            name: 识别到的人名
        返回：
            bool: 是否成功添加
        """
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with self._get_connection() as conn:
            try:
                cursor = conn.cursor()
                # 直接插入，依赖PRIMARY KEY处理重复
                cursor.execute(
                    "INSERT OR IGNORE INTO recognition_records (name, time) VALUES (?, ?)",
                    (name, current_time)
                )
                # 检查是否实际插入了数据
                return cursor.rowcount > 0
            except sqlite3.Error as e:
                print(f"数据库错误: {e}")
                return False
    
    def get_all_records(self):
        """获取所有记录，按时间降序排列"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name, time FROM recognition_records 
                ORDER BY time DESC
            """)
            columns = [column[0] for column in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_records_by_name(self, name):
        """按名称查询记录，按时间降序排列"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name, time FROM recognition_records 
                WHERE name = ? 
                ORDER BY time DESC
            """, (name,))
            columns = [column[0] for column in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_records_by_time_range(self, start_time, end_time):
        """按时间范围查询记录，按时间降序排列"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name, time FROM recognition_records 
                WHERE time BETWEEN ? AND ? 
                ORDER BY time ASC
            """, (start_time, end_time))
                # ORDER BY time DESC
            columns = [column[0] for column in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        
    def delete_records_by_name(self, name):
        """
        删除指定 name 的所有识别记录

        参数：
            name: 要删除记录的姓名
        返回：
            int: 删除的记录数量
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    DELETE FROM recognition_records
                    WHERE name = ?
                """, (name,))
                return cursor.rowcount  # 返回删除的记录数
            except sqlite3.Error as e:
                print(f"删除失败: {e}")
                return 0
    
    def reset_database(self):
        """重置数据库（清空所有记录）"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS recognition_records")
            self._create_table()  # 重新创建表
    
    def close_all(self):
        """关闭所有数据库连接"""
        while not self._connection_pool.empty():
            conn = self._connection_pool.get()
            conn.close()

# 测试代码
if __name__ == "__main__":
    db = RecognitionDatabase()
    try:
        # 查询结果，测试验证
        print("\n所有记录:")
        for record in db.get_all_records():
            print(record)

        deleted_count = db.delete_records_by_name("szh")
        print(f"已删除 {deleted_count} 条记录")

        print("\n所有记录:")
        for record in db.get_all_records():
            print(record)
    finally:
        db.close_all()