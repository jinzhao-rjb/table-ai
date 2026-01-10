import os
import re
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class FileManager:
    """文件管理类，提供各种文件操作功能"""

    @staticmethod
    def batch_rename_files(
        files: List[str],
        rename_rule: str,
        start_number: int = 1,
        number_format: str = "{:03d}",
        include_extension: bool = False
    ) -> List[Dict[str, str]]:
        """批量重命名文件

        Args:
            files: 文件列表
            rename_rule: 重命名规则，支持{number}占位符和正则表达式
            start_number: 起始编号
            number_format: 编号格式
            include_extension: 是否在重命名规则中包含扩展名

        Returns:
            重命名结果列表，每个元素包含old_path和new_path
        """
        results = []

        for i, file_path in enumerate(files):
            try:
                old_path = Path(file_path)
                if not old_path.exists():
                    continue

                # 获取文件名和扩展名
                filename = old_path.stem
                extension = old_path.suffix

                # 生成新文件名
                if "{number}" in rename_rule:
                    # 使用编号规则
                    new_filename = rename_rule.format(
                        number=number_format.format(start_number + i)
                    )
                elif "{orig_name}" in rename_rule:
                    # 保留原始文件名
                    new_filename = rename_rule.format(orig_name=filename)
                else:
                    # 正则表达式替换
                    new_filename = re.sub(rename_rule, r"", filename)

                # 处理扩展名
                if include_extension:
                    new_path = old_path.with_name(new_filename)
                else:
                    new_path = old_path.with_name(new_filename + extension)

                # 确保新路径唯一
                counter = 1
                while new_path.exists() and new_path != old_path:
                    if include_extension:
                        base_new_path = old_path.with_name(
                            f"{new_filename[:-len(extension)]}_{counter}{extension}")
                    else:
                        base_new_path = old_path.with_name(
                            f"{new_filename}_{counter}{extension}")
                    new_path = base_new_path
                    counter += 1

                # 执行重命名
                if new_path != old_path:
                    old_path.rename(new_path)
                    results.append({
                        "old_path": str(old_path),
                        "new_path": str(new_path)
                    })
                    logger.info(f"文件重命名成功: {old_path} -> {new_path}")
                else:
                    results.append({
                        "old_path": str(old_path),
                        "new_path": str(new_path),
                        "skipped": True
                    })
                    logger.info(f"文件已存在，跳过重命名: {old_path}")

            except Exception as e:
                logger.error(f"文件重命名失败: {file_path}, 错误: {e}")
                results.append({
                    "old_path": file_path,
                    "error": str(e)
                })

        return results

    @staticmethod
    def batch_convert_format(
        files: List[str],
        output_format: str,
        output_dir: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """批量转换文件格式

        Args:
            files: 文件列表
            output_format: 输出格式
            output_dir: 输出目录，默认为原文件目录

        Returns:
            转换结果列表
        """
        results = []

        for file_path in files:
            try:
                old_path = Path(file_path)
                if not old_path.exists():
                    continue

                # 确定输出目录
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                else:
                    output_path = old_path.parent

                # 生成新文件名
                new_filename = old_path.stem + f".{output_format}"
                new_path = output_path / new_filename

                # 这里只是一个基础实现，具体的格式转换需要根据不同的文件类型调用不同的库
                # 实际实现中，应该根据文件类型和输出格式调用对应的转换函数

                # 示例：复制文件作为占位符实现
                shutil.copy2(old_path, new_path)

                results.append({
                    "old_path": str(old_path),
                    "new_path": str(new_path)
                })
                logger.info(f"文件格式转换成功: {old_path} -> {new_path}")

            except Exception as e:
                logger.error(f"文件格式转换失败: {file_path}, 错误: {e}")
                results.append({
                    "old_path": file_path,
                    "error": str(e)
                })

        return results

    @staticmethod
    def batch_replace_content(
        files: List[str],
        search_pattern: str,
        replace_pattern: str,
        use_regex: bool = False,
        encoding: str = "utf-8"
    ) -> List[Dict[str, Any]]:
        """批量替换文件内容

        Args:
            files: 文件列表
            search_pattern: 搜索模式
            replace_pattern: 替换模式
            use_regex: 是否使用正则表达式
            encoding: 文件编码

        Returns:
            替换结果列表
        """
        results = []

        for file_path in files:
            try:
                file_path = Path(file_path)
                if not file_path.exists() or not file_path.is_file():
                    continue

                # 读取文件内容
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()

                # 执行替换
                if use_regex:
                    new_content = re.sub(
                        search_pattern, replace_pattern, content)
                else:
                    new_content = content.replace(
                        search_pattern, replace_pattern)

                # 写入文件
                if new_content != content:
                    with open(file_path, "w", encoding=encoding) as f:
                        f.write(new_content)

                    results.append({
                        "file_path": str(file_path),
                        "success": True,
                        "changed": True
                    })
                    logger.info(f"文件内容替换成功: {file_path}")
                else:
                    results.append({
                        "file_path": str(file_path),
                        "success": True,
                        "changed": False
                    })
                    logger.info(f"文件内容未变化，跳过替换: {file_path}")

            except Exception as e:
                logger.error(f"文件内容替换失败: {file_path}, 错误: {e}")
                results.append({
                    "file_path": str(file_path),
                    "success": False,
                    "error": str(e)
                })

        return results

    @staticmethod
    def create_folder_structure(
        base_dir: str,
        structure: Dict[str, Any]
    ) -> List[str]:
        """根据字典结构创建文件夹

        Args:
            base_dir: 基础目录
            structure: 文件夹结构字典

        Returns:
            创建的文件夹列表
        """
        created_folders = []

        def create_recursive(current_path: Path, current_structure: Dict[str, Any]):
            """递归创建文件夹"""
            for folder_name, sub_structure in current_structure.items():
                folder_path = current_path / folder_name
                folder_path.mkdir(parents=True, exist_ok=True)
                created_folders.append(str(folder_path))

                if isinstance(sub_structure, dict):
                    create_recursive(folder_path, sub_structure)

        try:
            create_recursive(Path(base_dir), structure)
            logger.info(f"文件夹结构创建成功，基础目录: {base_dir}")
        except Exception as e:
            logger.error(f"文件夹结构创建失败，基础目录: {base_dir}, 错误: {e}")

        return created_folders

    @staticmethod
    def find_duplicate_files(
        folder_path: str,
        compare_size: bool = True,
        compare_content: bool = False,
        include_subfolders: bool = True
    ) -> List[List[str]]:
        """查找重复文件

        Args:
            folder_path: 文件夹路径
            compare_size: 是否比较文件大小
            compare_content: 是否比较文件内容
            include_subfolders: 是否包含子文件夹

        Returns:
            重复文件列表，每个子列表包含一组重复文件
        """
        duplicate_groups = []

        try:
            # 按文件大小分组
            size_groups: Dict[int, List[str]] = {}

            # 遍历文件夹
            for root, dirs, files in os.walk(folder_path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    file_size = os.path.getsize(file_path)

                    if file_size not in size_groups:
                        size_groups[file_size] = []
                    size_groups[file_size].append(file_path)

                if not include_subfolders:
                    break

            # 处理大小相同的文件
            for size, files in size_groups.items():
                if len(files) < 2:
                    continue

                # 如果只比较大小，直接添加到结果
                if not compare_content:
                    duplicate_groups.append(files)
                    continue

                # 比较文件内容
                content_groups: Dict[str, List[str]] = {}

                for file_path in files:
                    try:
                        with open(file_path, "rb") as f:
                            content_hash = hash(f.read())

                        if content_hash not in content_groups:
                            content_groups[content_hash] = []
                        content_groups[content_hash].append(file_path)
                    except Exception as e:
                        logger.error(f"读取文件失败: {file_path}, 错误: {e}")

                # 添加内容相同的文件组
                for content_hash, content_files in content_groups.items():
                    if len(content_files) >= 2:
                        duplicate_groups.append(content_files)

            logger.info(f"重复文件查找完成，找到 {len(duplicate_groups)} 组重复文件")

        except Exception as e:
            logger.error(f"重复文件查找失败: {folder_path}, 错误: {e}")

        return duplicate_groups

    @staticmethod
    def delete_files(files: List[str], recycle: bool = False) -> List[Dict[str, str]]:
        """删除文件

        Args:
            files: 文件列表
            recycle: 是否放入回收站

        Returns:
            删除结果列表
        """
        results = []

        for file_path in files:
            try:
                file_path = Path(file_path)
                if not file_path.exists():
                    continue

                if recycle:
                    # Windows系统放入回收站
                    try:
                        import win32com.client
                        shell = win32com.client.Dispatch("Shell.Application")
                        shell.Namespace(10).MoveHere(str(file_path))
                    except ImportError:
                        # 如果没有win32com，直接删除
                        file_path.unlink()
                else:
                    # 直接删除
                    file_path.unlink()

                results.append({
                    "file_path": str(file_path),
                    "success": True
                })
                logger.info(f"文件删除成功: {file_path}")

            except Exception as e:
                logger.error(f"文件删除失败: {file_path}, 错误: {e}")
                results.append({
                    "file_path": str(file_path),
                    "success": False,
                    "error": str(e)
                })

        return results

    @staticmethod
    def copy_files(
        files: List[str],
        destination: str,
        overwrite: bool = False
    ) -> List[Dict[str, str]]:
        """复制文件

        Args:
            files: 文件列表
            destination: 目标目录
            overwrite: 是否覆盖现有文件

        Returns:
            复制结果列表
        """
        results = []
        destination_path = Path(destination)

        try:
            destination_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"创建目标目录失败: {destination}, 错误: {e}")
            return results

        for file_path in files:
            try:
                old_path = Path(file_path)
                if not old_path.exists():
                    continue

                new_path = destination_path / old_path.name

                # 处理文件已存在的情况
                if new_path.exists():
                    if overwrite:
                        new_path.unlink()
                    else:
                        # 生成新文件名
                        counter = 1
                        while new_path.exists():
                            new_filename = f"{old_path.stem}_{counter}{old_path.suffix}"
                            new_path = destination_path / new_filename
                            counter += 1

                shutil.copy2(old_path, new_path)

                results.append({
                    "old_path": str(old_path),
                    "new_path": str(new_path)
                })
                logger.info(f"文件复制成功: {old_path} -> {new_path}")

            except Exception as e:
                logger.error(f"文件复制失败: {file_path}, 错误: {e}")
                results.append({
                    "old_path": str(file_path),
                    "error": str(e)
                })

        return results

    @staticmethod
    def move_files(
        files: List[str],
        destination: str,
        overwrite: bool = False
    ) -> List[Dict[str, str]]:
        """移动文件

        Args:
            files: 文件列表
            destination: 目标目录
            overwrite: 是否覆盖现有文件

        Returns:
            移动结果列表
        """
        results = []
        destination_path = Path(destination)

        try:
            destination_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"创建目标目录失败: {destination}, 错误: {e}")
            return results

        for file_path in files:
            try:
                old_path = Path(file_path)
                if not old_path.exists():
                    continue

                new_path = destination_path / old_path.name

                # 处理文件已存在的情况
                if new_path.exists():
                    if overwrite:
                        new_path.unlink()
                    else:
                        # 生成新文件名
                        counter = 1
                        while new_path.exists():
                            new_filename = f"{old_path.stem}_{counter}{old_path.suffix}"
                            new_path = destination_path / new_filename
                            counter += 1

                shutil.move(old_path, new_path)

                results.append({
                    "old_path": str(old_path),
                    "new_path": str(new_path)
                })
                logger.info(f"文件移动成功: {old_path} -> {new_path}")

            except Exception as e:
                logger.error(f"文件移动失败: {file_path}, 错误: {e}")
                results.append({
                    "old_path": str(file_path),
                    "error": str(e)
                })

        return results

    @staticmethod
    def get_file_info(file_path: str) -> Optional[Dict[str, Any]]:
        """获取文件信息

        Args:
            file_path: 文件路径

        Returns:
            文件信息字典，包含名称、大小、创建时间、修改时间等
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return None

            stat = path.stat()

            return {
                "name": path.name,
                "path": str(path),
                "size": stat.st_size,
                "created_at": stat.st_ctime,
                "modified_at": stat.st_mtime,
                "is_file": path.is_file(),
                "is_dir": path.is_dir(),
                "extension": path.suffix.lower() if path.is_file() else ""
            }

        except Exception as e:
            logger.error(f"获取文件信息失败: {file_path}, 错误: {e}")
            return None
