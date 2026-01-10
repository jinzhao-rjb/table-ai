import logging
from PyQt5.QtCore import QObject, QThread, QRunnable, QThreadPool, pyqtSignal
from typing import Callable, Any, Dict

logger = logging.getLogger(__name__)


class AsyncTaskWorker(QRunnable):
    """异步任务工作类，继承自QRunnable

    用于在后台线程执行耗时任务，并发送进度和结果信号
    """

    class Signals(QObject):
        """信号定义类"""
        finished = pyqtSignal(object)  # 任务完成信号，传递结果
        error = pyqtSignal(Exception)  # 错误信号，传递异常
        progress = pyqtSignal(int, str)  # 进度信号，传递进度百分比和状态描述
        status = pyqtSignal(str)  # 状态信号，传递状态描述

    def __init__(self, task_func: Callable, *args, **kwargs):
        """初始化异步任务工作类

        Args:
            task_func: 要执行的任务函数
            *args: 任务函数的位置参数
            **kwargs: 任务函数的关键字参数
        """
        super().__init__()
        self.signals = AsyncTaskWorker.Signals()
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs
        self.is_cancelled = False

        # 添加进度回调到kwargs
        if 'progress_callback' not in self.kwargs:
            self.kwargs['progress_callback'] = self.progress_callback

        # 添加状态回调到kwargs
        if 'status_callback' not in self.kwargs:
            self.kwargs['status_callback'] = self.status_callback

    def progress_callback(self, percentage: int, status: str = ""):
        """进度回调函数

        Args:
            percentage: 进度百分比（0-100）
            status: 状态描述
        """
        if not self.is_cancelled:
            self.signals.progress.emit(percentage, status)

    def status_callback(self, status: str):
        """状态回调函数

        Args:
            status: 状态描述
        """
        if not self.is_cancelled:
            self.signals.status.emit(status)

    def cancel(self):
        """取消任务"""
        self.is_cancelled = True
        self.signals.status.emit("任务已取消")

    def run(self):
        """执行任务"""
        try:
            logger.debug(f"开始执行异步任务: {self.task_func.__name__}")
            result = self.task_func(*self.args, **self.kwargs)
            if not self.is_cancelled:
                logger.debug(f"异步任务执行成功: {self.task_func.__name__}")
                self.signals.finished.emit(result)
        except Exception as e:
            if not self.is_cancelled:
                logger.error(f"异步任务执行失败: {self.task_func.__name__}, 错误: {e}")
                self.signals.error.emit(e)
        finally:
            # 清理资源
            self.args = None
            self.kwargs = None


class AsyncTaskManager(QObject):
    """异步任务管理器

    用于管理和执行异步任务，提供线程池管理和任务控制
    """

    def __init__(self, max_threads: int = 4):
        """初始化异步任务管理器

        Args:
            max_threads: 最大线程数，默认4
        """
        super().__init__()
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(max_threads)
        self.active_tasks: Dict[str, AsyncTaskWorker] = {}

        logger.info(f"异步任务管理器初始化完成，最大线程数: {max_threads}")

    def execute_task(self, task_name: str, task_func: Callable, *args, **kwargs) -> AsyncTaskWorker:
        """执行异步任务

        Args:
            task_name: 任务名称，用于标识任务
            task_func: 要执行的任务函数
            *args: 任务函数的位置参数
            **kwargs: 任务函数的关键字参数

        Returns:
            AsyncTaskWorker: 异步任务工作对象
        """
        # 创建任务工作对象
        worker = AsyncTaskWorker(task_func, *args, **kwargs)
        self.active_tasks[task_name] = worker

        # 执行任务
        self.thread_pool.start(worker)
        logger.info(f"启动异步任务: {task_name}")

        return worker

    def cancel_task(self, task_name: str):
        """取消指定任务

        Args:
            task_name: 任务名称
        """
        if task_name in self.active_tasks:
            worker = self.active_tasks[task_name]
            worker.cancel()
            del self.active_tasks[task_name]
            logger.info(f"取消异步任务: {task_name}")

    def cancel_all_tasks(self):
        """取消所有任务"""
        for task_name in list(self.active_tasks.keys()):
            self.cancel_task(task_name)
        logger.info("取消所有异步任务")

    def get_active_task_count(self) -> int:
        """获取活动任务数

        Returns:
            int: 活动任务数
        """
        return len(self.active_tasks)

    def is_task_running(self, task_name: str) -> bool:
        """检查任务是否正在运行

        Args:
            task_name: 任务名称

        Returns:
            bool: True表示任务正在运行，False表示任务未运行
        """
        return task_name in self.active_tasks


# 创建全局异步任务管理器实例
async_task_manager = AsyncTaskManager(max_threads=4)
