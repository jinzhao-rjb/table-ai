import os
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import jieba
import jieba.analyse
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
import markdown
from markdown.extensions.toc import TocExtension
from markdown.extensions.codehilite import CodeHiliteExtension
from markdown.extensions.fenced_code import FencedCodeExtension

logger = logging.getLogger(__name__)


class TextTools:
    """文本处理类，提供各种文本操作功能"""

    @staticmethod
    def extract_keywords(
        text: str,
        top_n: int = 10,
        language: str = "zh"
    ) -> List[Dict[str, Any]]:
        """提取文本关键词

        Args:
            text: 要提取关键词的文本
            top_n: 提取的关键词数量
            language: 语言类型，"zh"表示中文，"en"表示英文

        Returns:
            关键词列表，每个元素包含word和weight
        """
        try:
            if language == "zh":
                # 使用jieba提取中文关键词
                keywords = jieba.analyse.extract_tags(
                    text,
                    topK=top_n,
                    withWeight=True
                )
                return [{"word": word, "weight": weight} for word, weight in keywords]
            else:
                # 使用nltk提取英文关键词
                tokens = word_tokenize(text.lower())
                # 过滤停用词
                stop_words = set(
                    ["the", "a", "an", "in", "on", "at", "for", "of", "to", "and", "or", "but"])
                filtered_tokens = [
                    token for token in tokens if token.isalpha() and token not in stop_words]
                # 计算词频
                freq_dist = FreqDist(filtered_tokens)
                return [{"word": word, "weight": freq} for word, freq in freq_dist.most_common(top_n)]
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            return []

    @staticmethod
    def batch_extract_keywords(
        files: List[str],
        top_n: int = 10,
        language: str = "zh"
    ) -> List[Dict[str, Any]]:
        """批量提取文本文件关键词

        Args:
            files: 文件列表
            top_n: 提取的关键词数量
            language: 语言类型，"zh"表示中文，"en"表示英文

        Returns:
            每个文件的关键词提取结果
        """
        results = []

        for file_path in files:
            try:
                file_path = Path(file_path)
                if not file_path.exists() or not file_path.is_file():
                    continue

                # 读取文件内容
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # 提取关键词
                keywords = TextTools.extract_keywords(content, top_n, language)

                results.append({
                    "file_path": str(file_path),
                    "success": True,
                    "keywords": keywords
                })
                logger.info(f"文件关键词提取成功: {file_path}")
            except Exception as e:
                logger.error(f"文件关键词提取失败: {file_path}, 错误: {e}")
                results.append({
                    "file_path": str(file_path),
                    "success": False,
                    "error": str(e)
                })

        return results

    @staticmethod
    def text_summary(
        text: str,
        max_sentences: int = 5,
        language: str = "zh"
    ) -> str:
        """生成文本摘要

        Args:
            text: 要生成摘要的文本
            max_sentences: 摘要的最大句子数量
            language: 语言类型，"zh"表示中文，"en"表示英文

        Returns:
            生成的摘要文本
        """
        try:
            if language == "zh":
                # 使用jieba进行中文分词
                sentences = re.split(r'[。！？\n]+', text)
                sentences = [sentence.strip()
                             for sentence in sentences if sentence.strip()]

                # 简单的摘要生成算法：选择包含关键词最多的句子
                keywords = jieba.analyse.extract_tags(text, topK=20)
                keyword_set = set(keywords)

                # 计算每个句子的关键词密度
                sentence_scores = []
                for i, sentence in enumerate(sentences):
                    if not sentence:
                        continue
                    # 分词
                    words = jieba.lcut(sentence)
                    # 计算关键词数量
                    keyword_count = len(
                        [word for word in words if word in keyword_set])
                    # 计算关键词密度
                    score = keyword_count / max(1, len(words))
                    sentence_scores.append((score, i, sentence))

                # 选择得分最高的句子
                sentence_scores.sort(reverse=True, key=lambda x: x[0])
                selected_sentences = sorted(
                    sentence_scores[:max_sentences], key=lambda x: x[1])
                summary = "。".join(
                    [sentence for _, _, sentence in selected_sentences]) + "。"
                return summary
            else:
                # 使用nltk进行英文摘要
                sentences = sent_tokenize(text)

                # 简单的摘要生成算法：选择包含关键词最多的句子
                tokens = word_tokenize(text.lower())
                stop_words = set(
                    ["the", "a", "an", "in", "on", "at", "for", "of", "to", "and", "or", "but"])
                filtered_tokens = [
                    token for token in tokens if token.isalpha() and token not in stop_words]

                # 计算词频
                freq_dist = FreqDist(filtered_tokens)
                keywords = [word for word, _ in freq_dist.most_common(20)]
                keyword_set = set(keywords)

                # 计算每个句子的关键词密度
                sentence_scores = []
                for i, sentence in enumerate(sentences):
                    if not sentence:
                        continue
                    words = word_tokenize(sentence.lower())
                    keyword_count = len(
                        [word for word in words if word in keyword_set])
                    score = keyword_count / max(1, len(words))
                    sentence_scores.append((score, i, sentence))

                # 选择得分最高的句子
                sentence_scores.sort(reverse=True, key=lambda x: x[0])
                selected_sentences = sorted(
                    sentence_scores[:max_sentences], key=lambda x: x[1])
                summary = " ".join(
                    [sentence for _, _, sentence in selected_sentences])
                return summary
        except Exception as e:
            logger.error(f"文本摘要生成失败: {e}")
            return text[:200] + "..." if len(text) > 200 else text

    @staticmethod
    def count_words(
        text: str,
        language: str = "zh"
    ) -> Dict[str, int]:
        """统计文本字数

        Args:
            text: 要统计字数的文本
            language: 语言类型，"zh"表示中文，"en"表示英文

        Returns:
            字数统计结果，包含characters、words、sentences等
        """
        try:
            characters = len(text)
            characters_no_space = len(text.replace(" ", "").replace(
                "\t", "").replace("\n", "").replace("\r", ""))

            if language == "zh":
                # 中文分词
                words = jieba.lcut(text)
                word_count = len(words)
                # 中文句子分割
                sentences = re.split(r'[。！？\n]+', text)
                sentence_count = len([s for s in sentences if s.strip()])
            else:
                # 英文分词
                words = word_tokenize(text)
                word_count = len(words)
                # 英文句子分割
                sentences = sent_tokenize(text)
                sentence_count = len(sentences)

            return {
                "characters": characters,
                "characters_no_space": characters_no_space,
                "words": word_count,
                "sentences": sentence_count
            }
        except Exception as e:
            logger.error(f"文本字数统计失败: {e}")
            return {
                "characters": 0,
                "characters_no_space": 0,
                "words": 0,
                "sentences": 0
            }

    @staticmethod
    def convert_markdown_to_html(
        markdown_text: str,
        include_toc: bool = True
    ) -> str:
        """将Markdown转换为HTML

        Args:
            markdown_text: Markdown文本
            include_toc: 是否包含目录

        Returns:
            转换后的HTML文本
        """
        try:
            # 配置Markdown扩展
            extensions = [
                CodeHiliteExtension(),
                FencedCodeExtension()
            ]

            if include_toc:
                extensions.append(TocExtension())

            # 转换Markdown为HTML
            html = markdown.markdown(
                markdown_text,
                extensions=extensions
            )

            return html
        except Exception as e:
            logger.error(f"Markdown转换失败: {e}")
            return f"<p>Markdown转换失败: {e}</p>"

    @staticmethod
    def batch_process_texts(
        files: List[str],
        operations: List[str],
        output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """批量处理文本文件

        Args:
            files: 文件列表
            operations: 要执行的操作列表，可选值包括"extract_keywords", "summary", "count_words", "convert_markdown"
            output_dir: 输出目录，默认为原文件目录

        Returns:
            处理结果列表
        """
        results = []

        for file_path in files:
            try:
                file_path = Path(file_path)
                if not file_path.exists() or not file_path.is_file():
                    continue

                # 读取文件内容
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # 执行操作
                operation_results = {}

                if "extract_keywords" in operations:
                    operation_results["keywords"] = TextTools.extract_keywords(
                        content)

                if "summary" in operations:
                    operation_results["summary"] = TextTools.text_summary(
                        content)

                if "count_words" in operations:
                    operation_results["word_count"] = TextTools.count_words(
                        content)

                if "convert_markdown" in operations and file_path.suffix.lower() in [".md", ".markdown"]:
                    operation_results["html"] = TextTools.convert_markdown_to_html(
                        content)

                # 保存结果
                if output_dir:
                    output_path = Path(
                        output_dir) / f"{file_path.stem}_processed{file_path.suffix}"
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(str(operation_results))

                results.append({
                    "file_path": str(file_path),
                    "success": True,
                    "results": operation_results
                })
                logger.info(f"文本处理成功: {file_path}")
            except Exception as e:
                logger.error(f"文本处理失败: {file_path}, 错误: {e}")
                results.append({
                    "file_path": str(file_path),
                    "success": False,
                    "error": str(e)
                })

        return results

    @staticmethod
    def batch_convert_files(
        files: List[str],
        output_format: str,
        output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """批量转换文件格式

        Args:
            files: 文件列表
            output_format: 输出格式，可选值包括"txt", "md", "html"
            output_dir: 输出目录，默认为原文件目录

        Returns:
            转换结果列表
        """
        results = []

        for file_path in files:
            try:
                file_path = Path(file_path)
                if not file_path.exists() or not file_path.is_file():
                    continue

                # 确定输出路径
                if output_dir:
                    output_path = Path(output_dir) / \
                        f"{file_path.stem}.{output_format}"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    output_path = file_path.with_suffix(f".{output_format}")

                # 读取文件内容
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # 执行转换
                if output_format == "html" and file_path.suffix.lower() in [".md", ".markdown"]:
                    converted_content = TextTools.convert_markdown_to_html(
                        content)
                else:
                    # 其他格式转换，这里简单复制
                    converted_content = content

                # 保存结果
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(converted_content)

                results.append({
                    "file_path": str(file_path),
                    "output_path": str(output_path),
                    "success": True
                })
                logger.info(f"文件格式转换成功: {file_path} -> {output_path}")
            except Exception as e:
                logger.error(f"文件格式转换失败: {file_path}, 错误: {e}")
                results.append({
                    "file_path": str(file_path),
                    "success": False,
                    "error": str(e)
                })

        return results

    @staticmethod
    def find_and_replace(
        text: str,
        search_pattern: str,
        replace_pattern: str,
        use_regex: bool = False
    ) -> str:
        """查找并替换文本

        Args:
            text: 要处理的文本
            search_pattern: 搜索模式
            replace_pattern: 替换模式
            use_regex: 是否使用正则表达式

        Returns:
            处理后的文本
        """
        try:
            if use_regex:
                return re.sub(search_pattern, replace_pattern, text)
            else:
                return text.replace(search_pattern, replace_pattern)
        except Exception as e:
            logger.error(f"文本替换失败: {e}")
            return text

    @staticmethod
    def batch_find_and_replace(
        files: List[str],
        search_pattern: str,
        replace_pattern: str,
        use_regex: bool = False,
        output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """批量查找并替换文本

        Args:
            files: 文件列表
            search_pattern: 搜索模式
            replace_pattern: 替换模式
            use_regex: 是否使用正则表达式
            output_dir: 输出目录，默认为原文件目录

        Returns:
            替换结果列表
        """
        results = []

        for file_path in files:
            try:
                file_path = Path(file_path)
                if not file_path.exists() or not file_path.is_file():
                    continue

                # 确定输出路径
                if output_dir:
                    output_path = Path(output_dir) / file_path.name
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    output_path = file_path.with_name(
                        f"{file_path.stem}_replaced{file_path.suffix}")

                # 读取文件内容
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # 执行替换
                replaced_content = TextTools.find_and_replace(
                    content,
                    search_pattern,
                    replace_pattern,
                    use_regex
                )

                # 保存结果
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(replaced_content)

                results.append({
                    "file_path": str(file_path),
                    "output_path": str(output_path),
                    "success": True
                })
                logger.info(f"文本替换成功: {file_path} -> {output_path}")
            except Exception as e:
                logger.error(f"文本替换失败: {file_path}, 错误: {e}")
                results.append({
                    "file_path": str(file_path),
                    "success": False,
                    "error": str(e)
                })

        return results
