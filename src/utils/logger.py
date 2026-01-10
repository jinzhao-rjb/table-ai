import os
import logging
import logging.handlers
from datetime import datetime


class LoggerManager:
    """日志管理类"""

    def __init__(self):
        self.log_dir = os.path.join(
            os.path.expanduser("~"), ".office-lazy-tool", "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # 创建日志文件名
        log_file = os.path.join(
            self.log_dir, f"office-lazy-tool_{datetime.now().strftime('%Y-%m-%d')}.log")

        # 创建日志记录器
        self.logger = logging.getLogger("office-lazy-tool")
        self.logger.setLevel(logging.DEBUG)

        # 避免重复添加处理器
        if not self.logger.handlers:
            # 创建控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # 创建文件处理器（按大小分割）
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,  # 保留5个备份文件
                encoding="utf-8"
            )
            file_handler.setLevel(logging.DEBUG)

            # 创建日志格式
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
            )

            # 设置格式
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            # 添加处理器
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

    def get_logger(self):
        """获取日志记录器"""
        return self.logger


# 创建全局日志记录器实例
logger_manager = LoggerManager()
logger = logger_manager.get_logger()
