#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AST 代码安全检查器
用于检测 Python 代码中的潜在安全问题
"""

import ast
import logging
from typing import List, Dict, Any

logger = logging.getLogger("ASTSecurityChecker")


class ASTSecurityChecker:
    """
    AST 代码安全检查器
    检测 Python 代码中的潜在安全问题
    """
    
    # 危险函数列表
    DANGEROUS_FUNCTIONS = {
        # 文件系统操作
        "open": {"risk": "high", "desc": "文件打开操作，可能导致文件泄露或覆盖"},
        "file": {"risk": "high", "desc": "文件操作，可能导致文件泄露或覆盖"},
        "os.open": {"risk": "high", "desc": "文件打开操作，可能导致文件泄露或覆盖"},
        "os.read": {"risk": "high", "desc": "文件读取操作，可能导致文件泄露"},
        "os.write": {"risk": "high", "desc": "文件写入操作，可能导致文件覆盖"},
        "os.remove": {"risk": "high", "desc": "文件删除操作，可能导致文件丢失"},
        "os.unlink": {"risk": "high", "desc": "文件删除操作，可能导致文件丢失"},
        "os.rename": {"risk": "high", "desc": "文件重命名操作，可能导致文件覆盖"},
        "os.replace": {"risk": "high", "desc": "文件替换操作，可能导致文件覆盖"},
        "os.makedirs": {"risk": "medium", "desc": "目录创建操作，可能导致目录遍历"},
        "os.rmdir": {"risk": "high", "desc": "目录删除操作，可能导致目录丢失"},
        "os.removedirs": {"risk": "high", "desc": "目录删除操作，可能导致目录丢失"},
        "shutil.copy": {"risk": "medium", "desc": "文件复制操作，可能导致文件泄露"},
        "shutil.copy2": {"risk": "medium", "desc": "文件复制操作，可能导致文件泄露"},
        "shutil.copyfile": {"risk": "medium", "desc": "文件复制操作，可能导致文件泄露"},
        "shutil.copyfileobj": {"risk": "medium", "desc": "文件复制操作，可能导致文件泄露"},
        "shutil.move": {"risk": "high", "desc": "文件移动操作，可能导致文件覆盖"},
        "shutil.rmtree": {"risk": "high", "desc": "目录删除操作，可能导致目录丢失"},
        
        # 命令执行
        "os.system": {"risk": "high", "desc": "系统命令执行，可能导致命令注入"},
        "subprocess.run": {"risk": "high", "desc": "子进程执行，可能导致命令注入"},
        "subprocess.call": {"risk": "high", "desc": "子进程执行，可能导致命令注入"},
        "subprocess.check_call": {"risk": "high", "desc": "子进程执行，可能导致命令注入"},
        "subprocess.check_output": {"risk": "high", "desc": "子进程执行，可能导致命令注入"},
        "subprocess.Popen": {"risk": "high", "desc": "子进程执行，可能导致命令注入"},
        "exec": {"risk": "high", "desc": "动态代码执行，可能导致代码注入"},
        "eval": {"risk": "high", "desc": "动态代码执行，可能导致代码注入"},
        "compile": {"risk": "medium", "desc": "动态代码编译，可能导致代码注入"},
        
        # 网络操作
        "socket.socket": {"risk": "medium", "desc": "网络套接字创建，可能导致网络连接"},
        "urllib.request.urlopen": {"risk": "medium", "desc": "URL打开操作，可能导致网络请求"},
        "requests.get": {"risk": "medium", "desc": "HTTP GET请求，可能导致网络请求"},
        "requests.post": {"risk": "medium", "desc": "HTTP POST请求，可能导致网络请求"},
        "requests.put": {"risk": "medium", "desc": "HTTP PUT请求，可能导致网络请求"},
        "requests.delete": {"risk": "medium", "desc": "HTTP DELETE请求，可能导致网络请求"},
        "http.client.HTTPConnection": {"risk": "medium", "desc": "HTTP连接创建，可能导致网络请求"},
        "http.client.HTTPSConnection": {"risk": "medium", "desc": "HTTPS连接创建，可能导致网络请求"},
        
        # 环境变量
        "os.environ": {"risk": "medium", "desc": "环境变量访问，可能导致敏感信息泄露"},
        "os.getenv": {"risk": "medium", "desc": "环境变量访问，可能导致敏感信息泄露"},
        "os.putenv": {"risk": "medium", "desc": "环境变量设置，可能导致环境污染"},
        
        # 进程操作
        "os.fork": {"risk": "medium", "desc": "进程创建，可能导致资源耗尽"},
        "os.kill": {"risk": "high", "desc": "进程终止，可能导致进程崩溃"},
        "os.killpg": {"risk": "high", "desc": "进程组终止，可能导致进程崩溃"},
        "os.wait": {"risk": "medium", "desc": "进程等待，可能导致死锁"},
        "os.waitpid": {"risk": "medium", "desc": "进程等待，可能导致死锁"},
        "os.wait3": {"risk": "medium", "desc": "进程等待，可能导致死锁"},
        "os.wait4": {"risk": "medium", "desc": "进程等待，可能导致死锁"},
    }
    
    # 危险模块列表
    DANGEROUS_MODULES = {
        "os": {"risk": "high", "desc": "操作系统接口，包含多种危险操作"},
        "sys": {"risk": "medium", "desc": "系统模块，可能导致系统级操作"},
        "subprocess": {"risk": "high", "desc": "子进程管理，可能导致命令注入"},
        "shutil": {"risk": "high", "desc": "高级文件操作，可能导致文件系统破坏"},
        "socket": {"risk": "medium", "desc": "网络套接字，可能导致网络连接"},
        "urllib": {"risk": "medium", "desc": "URL处理，可能导致网络请求"},
        "urllib2": {"risk": "medium", "desc": "URL处理，可能导致网络请求"},
        "requests": {"risk": "medium", "desc": "HTTP库，可能导致网络请求"},
        "http": {"risk": "medium", "desc": "HTTP库，可能导致网络请求"},
        "httplib": {"risk": "medium", "desc": "HTTP库，可能导致网络请求"},
        "ctypes": {"risk": "high", "desc": "外部库调用，可能导致系统级操作"},
        "pickle": {"risk": "high", "desc": "序列化库，可能导致反序列化漏洞"},
        "cPickle": {"risk": "high", "desc": "序列化库，可能导致反序列化漏洞"},
        "marshal": {"risk": "high", "desc": "序列化库，可能导致代码执行"},
        "shelve": {"risk": "medium", "desc": "持久化存储，可能导致文件操作"},
    }
    
    def __init__(self):
        """
        初始化AST安全检查器
        """
        self.issues = []
        self.current_line = 0
    
    def _add_issue(self, line: int, risk: str, desc: str, code: str):
        """
        添加安全问题
        
        Args:
            line: 问题所在行
            risk: 风险级别 (high, medium, low)
            desc: 问题描述
            code: 问题代码
        """
        self.issues.append({
            "line": line,
            "risk": risk,
            "desc": desc,
            "code": code
        })
    
    def _check_import(self, node: ast.Import):
        """
        检查导入语句
        
        Args:
            node: Import节点
        """
        for name in node.names:
            if name.name in self.DANGEROUS_MODULES:
                risk_info = self.DANGEROUS_MODULES[name.name]
                self._add_issue(
                    node.lineno,
                    risk_info["risk"],
                    f"导入危险模块: {risk_info['desc']}",
                    f"import {name.name}"
                )
    
    def _check_import_from(self, node: ast.ImportFrom):
        """
        检查从模块导入语句
        
        Args:
            node: ImportFrom节点
        """
        if node.module in self.DANGEROUS_MODULES:
            risk_info = self.DANGEROUS_MODULES[node.module]
            for name in node.names:
                self._add_issue(
                    node.lineno,
                    risk_info["risk"],
                    f"从危险模块导入: {risk_info['desc']}",
                    f"from {node.module} import {name.name}"
                )
    
    def _check_call(self, node: ast.Call):
        """
        检查函数调用
        
        Args:
            node: Call节点
        """
        # 构建函数名
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # 处理属性访问，如 os.open
            attr_path = []
            current = node.func
            while isinstance(current, ast.Attribute):
                attr_path.insert(0, current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                attr_path.insert(0, current.id)
                func_name = ".".join(attr_path)
        
        if func_name in self.DANGEROUS_FUNCTIONS:
            risk_info = self.DANGEROUS_FUNCTIONS[func_name]
            # 获取调用代码
            code = self._get_node_code(node)
            self._add_issue(
                node.lineno,
                risk_info["risk"],
                f"调用危险函数: {risk_info['desc']}",
                code
            )
    
    def _check_exec(self, node: ast.Expr):
        """
        检查exec语句
        
        Args:
            node: Expr节点
        """
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "exec":
            self._add_issue(
                node.lineno,
                "high",
                "使用exec函数，可能导致代码注入",
                "exec(...)")
    
    def _check_eval(self, node: ast.Expr):
        """
        检查eval语句
        
        Args:
            node: Expr节点
        """
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "eval":
            self._add_issue(
                node.lineno,
                "high",
                "使用eval函数，可能导致代码注入",
                "eval(...)")
    
    def _get_node_code(self, node: ast.AST) -> str:
        """
        获取节点对应的代码字符串
        
        Args:
            node: AST节点
            
        Returns:
            str: 代码字符串
        """
        # 简化实现，返回节点类型
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return f"{node.func.id}(...)"
            elif isinstance(node.func, ast.Attribute):
                return f"{self._get_node_code(node.func)}(...)"
        return str(type(node).__name__)
    
    def visit(self, node: ast.AST):
        """
        访问AST节点
        
        Args:
            node: AST节点
        """
        # 更新当前行
        if hasattr(node, 'lineno'):
            self.current_line = node.lineno
        
        # 根据节点类型调用相应的检查方法
        if isinstance(node, ast.Import):
            self._check_import(node)
        elif isinstance(node, ast.ImportFrom):
            self._check_import_from(node)
        elif isinstance(node, ast.Call):
            self._check_call(node)
        elif isinstance(node, ast.Expr):
            self._check_exec(node)
            self._check_eval(node)
        
        # 递归访问子节点
        for child in ast.iter_child_nodes(node):
            self.visit(child)
    
    def check_code(self, code: str) -> List[Dict[str, Any]]:
        """
        检查代码的安全性
        
        Args:
            code: 要检查的Python代码
            
        Returns:
            List[Dict[str, Any]]: 安全问题列表
        """
        self.issues = []
        try:
            # 解析代码为AST
            tree = ast.parse(code)
            # 访问所有节点
            self.visit(tree)
            return self.issues
        except SyntaxError as e:
            logger.error(f"代码语法错误: {e}")
            self._add_issue(
                e.lineno if hasattr(e, 'lineno') else 0,
                "high",
                f"代码语法错误: {e}",
                ""
            )
            return self.issues
        except Exception as e:
            logger.error(f"AST解析错误: {e}")
            self._add_issue(
                0,
                "high",
                f"AST解析错误: {e}",
                ""
            )
            return self.issues
    
    def is_safe(self, code: str) -> bool:
        """
        检查代码是否安全
        
        Args:
            code: 要检查的Python代码
            
        Returns:
            bool: 安全返回True，否则返回False
        """
        issues = self.check_code(code)
        # 只检查高风险问题
        high_risk_issues = [issue for issue in issues if issue["risk"] == "high"]
        return len(high_risk_issues) == 0
    
    def get_security_report(self, code: str) -> Dict[str, Any]:
        """
        获取安全检查报告
        
        Args:
            code: 要检查的Python代码
            
        Returns:
            Dict[str, Any]: 安全检查报告
        """
        issues = self.check_code(code)
        
        # 统计风险级别
        risk_counts = {
            "high": 0,
            "medium": 0,
            "low": 0
        }
        
        for issue in issues:
            if issue["risk"] in risk_counts:
                risk_counts[issue["risk"]] += 1
        
        return {
            "issues": issues,
            "risk_counts": risk_counts,
            "is_safe": len([i for i in issues if i["risk"] == "high"]) == 0,
            "total_issues": len(issues)
        }


def test_security_checker():
    """
    测试AST安全检查器
    """
    checker = ASTSecurityChecker()
    
    # 测试危险代码
    dangerous_code = """
import os
import subprocess

def dangerous_func():
    # 文件操作
    f = open("/etc/passwd", "r")
    content = f.read()
    f.close()
    
    # 命令执行
    os.system("ls -la")
    subprocess.run(["rm", "-rf", "/tmp/test"])
    
    # 动态代码执行
    eval("print('test')")
    exec("import sys; sys.exit(1)")
    """
    
    report = checker.get_security_report(dangerous_code)
    print("=== 安全检查报告 ===")
    print(f"总问题数: {report['total_issues']}")
    print(f"风险级别统计: {report['risk_counts']}")
    print(f"代码是否安全: {'是' if report['is_safe'] else '否'}")
    
    if report['issues']:
        print("\n=== 问题详情 ===")
        for issue in report['issues']:
            print(f"行 {issue['line']} [{issue['risk']}]: {issue['desc']}")
            print(f"  代码: {issue['code']}")


if __name__ == "__main__":
    test_security_checker()