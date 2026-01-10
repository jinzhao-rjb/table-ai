#!/usr/bin/env python3
"""
AI提示词生成器
负责根据用户需求和数据上下文生成优化的提示词
"""

import json
from typing import Dict, Any, List


class PromptGenerator:
    """
    提示词生成器
    负责根据用户需求和数据上下文生成优化的提示词
    """
    
    def __init__(self):
        self.logger = None  # 可选的日志记录器
    
    def generate_prompt(self, requirement: str, data_context: Dict[str, Any]) -> str:
        """
        生成优化的提示词
        
        Args:
            requirement: 用户需求
            data_context: 数据上下文
            
        Returns:
            优化后的提示词
        """
        # 分析需求中的依赖关系
        dependency_analysis = self._analyze_dependencies(requirement)
        
        # 构建增强的提示词
        enhanced_prompt = self._build_enhanced_prompt(requirement, data_context, dependency_analysis)
        
        if self.logger:
            self.logger.info(f"生成提示词: {enhanced_prompt[:200]}...")
        
        return enhanced_prompt
    
    def _analyze_dependencies(self, requirement: str) -> Dict[str, Any]:
        """
        分析需求中的依赖关系
        
        Args:
            requirement: 用户需求
            
        Returns:
            依赖关系分析结果
        """
        dependencies = []
        steps = []
        
        # 通用需求分解 - 识别所有可能的步骤
        lines = requirement.split('\n')
        current_step = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 识别任何包含冒号的步骤定义
            if '：' in line:
                # 分割步骤名和描述
                parts = line.split('：', 1)
                if len(parts) >= 2:
                    step_name = parts[0].strip()
                    step_desc = parts[1].strip()
                    
                    step = {
                        'step_name': step_name,
                        'description': step_desc,
                        'dependencies': []
                    }
                    
                    # 检查是否包含常见的依赖关键词
                    step_lower = step_desc.lower()
                    
                    # 识别各种依赖关系
                    if '依赖' in step_desc or '基于' in step_desc or '根据' in step_desc:
                        # 尝试从描述中提取依赖关系
                        import re
                        # 匹配 "基于..." "依赖..." "使用..." 等模式
                        patterns = [
                            r'基于\s*([^，。]+)',
                            r'依赖\s*([^，。]+)',
                            r'使用\s*([^，。]+)',
                            r'([^，。]*)\s*计算',
                        ]
                        
                        for pattern in patterns:
                            matches = re.findall(pattern, step_desc)
                            for match in matches:
                                if match.strip() and match.strip() != step_name:
                                    # 提取可能的依赖列名
                                    potential_deps = [word.strip() for word in match.split() if word.strip()]
                                    step['dependencies'].extend(potential_deps)
                    
                    steps.append(step)
        
        # 通用依赖分析 - 分析所有步骤之间的潜在依赖
        for i, step in enumerate(steps):
            step_desc_lower = step['description'].lower()
            
            # 检查是否引用了前面步骤的结果
            for j, prev_step in enumerate(steps[:i]):
                prev_step_name = prev_step['step_name']
                if prev_step_name in step_desc_lower:
                    if prev_step_name not in step['dependencies']:
                        step['dependencies'].append(prev_step_name)
        
        # 特殊处理：识别常见的数学运算依赖关系
        for step in steps:
            desc = step['description']
            # 识别 "A = B * C" 或 "A由B和C计算" 等模式
            import re
            # 匹配 "结果 = 因素1 * 因素2" 模式
            calc_patterns = [
                r'([^=]+)=([^*+\-/]+)[*+\-/]([^,，。]+)',
                r'([^由]+)由([^和]+)和([^计算]+)计算',
                r'([^基于]+)基于([^和]+)和([^计算]+)',
            ]
            
            for pattern in calc_patterns:
                matches = re.findall(pattern, desc)
                for match in matches:
                    if len(match) >= 2:
                        # 假设右侧是依赖项
                        for item in match[1:]:
                            clean_item = item.strip().replace('的', '').replace('值', '')
                            if clean_item and clean_item not in step['dependencies']:
                                step['dependencies'].append(clean_item)
        
        # 识别链式依赖
        for line in lines:
            line_lower = line.lower()
            
            # 识别链式依赖
            if '依赖' in line or '链式' in line or '基于' in line:
                dependencies.append({
                    'type': 'chain_dependency',
                    'description': line.strip()
                })
            
            # 识别计算顺序
            if '→' in line or '->' in line or '然后' in line or '再' in line:
                dependencies.append({
                    'type': 'execution_order',
                    'description': line.strip()
                })
        
        # 识别特定的依赖关系
        if '健康值' in requirement and '环境修正' in requirement:
            dependencies.append({
                'type': 'specific_dependency',
                'from': '环境修正指数',
                'to': '实时健康值',
                'description': '实时健康值依赖环境修正指数计算'
            })
        
        if '风险评定' in requirement and ('健康值' in requirement or '补给' in requirement):
            dependencies.append({
                'type': 'specific_dependency',
                'from': '实时健康值和加班补给逻辑',
                'to': '最终风险评定',
                'description': '最终风险评定依赖实时健康值和加班补给量'
            })
        
        return {
            'count': len(dependencies),
            'dependencies': dependencies,
            'steps': steps,
            'has_chain_dependency': any(dep['type'] == 'chain_dependency' for dep in dependencies),
            'has_steps': len(steps) > 0
        }
    
    def _build_enhanced_prompt(self, requirement: str, data_context: Dict[str, Any], dependency_analysis: Dict[str, Any]) -> str:
        """
        构建增强提示词
        
        Args:
            requirement: 原始需求
            data_context: 数据上下文
            dependency_analysis: 依赖关系分析
            
        Returns:
            增强后的提示词
        """
        # 基础提示词模板
        base_prompt = f"""
你是一个专业的数据处理AI，负责根据用户需求生成数据处理函数。

数据上下文:
- 列名: {data_context.get('columns', [])}
- 数据类型: {data_context.get('data_types', {})}
- 数据形状: {data_context.get('data_shape', (0, 0))}

原始需求:
{requirement}

"""
        
        # 添加需求分解提示
        if dependency_analysis['has_steps']:
            base_prompt += f"""
需求分解:
检测到以下处理步骤：
"""
            for i, step in enumerate(dependency_analysis['steps'], 1):
                base_prompt += f"\n{i}. {step['step_name']}: {step['description']}"
                if step['dependencies']:
                    base_prompt += f" (依赖: {', '.join(step['dependencies'])})"
        
        # 添加依赖关系提示
        if dependency_analysis['dependencies']:
            base_prompt += f"""

重要提示 - 函数依赖关系:
检测到以下依赖关系，必须按顺序处理：
"""
            for i, dep in enumerate(dependency_analysis['dependencies'], 1):
                base_prompt += f"\n{i}. {dep['description']}"
            
            base_prompt += """
执行顺序要求:
1. 首先生成所有基础计算列（不需要依赖其他新生成列的列）
2. 然后生成依赖于基础列的中间计算列
3. 最后生成依赖于前面所有列的最终列
4. 确保在使用任何新生成的列之前，它们已经被创建
"""
        
        # 添加步骤处理提示
        if dependency_analysis['has_steps']:
            base_prompt += """
处理步骤要求:
1. 将复杂需求分解为独立的处理步骤
2. 为每个步骤生成独立的函数
3. 确保函数按照依赖顺序执行
4. 每个函数专注于单一任务
"""
        
        # 添加技术要求
        base_prompt += """
技术要求:
1. 使用pandas进行数据处理，列名使用 df['列名'] 格式
2. 每个函数应返回完整的DataFrame
3. 确保列名使用正确的中文字符
4. 处理日期时使用适当的datetime函数
5. 生成的函数应包含适当的错误处理

输出格式:
请返回一个JSON数组，包含需要执行的函数列表，每个函数包含以下字段：
- name: 函数名
- description: 函数描述
- implementation: 函数实现代码
- required_columns: 依赖的列名列表
- new_columns: 生成的新列名列表
"""
        
        return base_prompt.strip()


# 单例模式
_prompt_generator_instance = None


def get_prompt_generator() -> PromptGenerator:
    """
    获取提示词生成器实例
    
    Returns:
        PromptGenerator实例
    """
    global _prompt_generator_instance
    if _prompt_generator_instance is None:
        _prompt_generator_instance = PromptGenerator()
    return _prompt_generator_instance