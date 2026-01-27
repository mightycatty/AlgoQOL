import ast
import os
from typing import List, Dict, Optional

def parse_file(filepath: str, rel_path: str) -> List[str]:
    """解析单个 Python 文件并返回 Markdown 格式的文档段落"""
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except SyntaxError:
            return []

    docs = []
    has_content = False

    # 获取模块级文档字符串
    module_doc = ast.get_docstring(tree)
    if module_doc:
        docs.append(f"### Module: `{rel_path}`\n")
        docs.append(f"{module_doc}\n")
        has_content = True

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            has_content = True
            if not module_doc: # 如果没有模块文档，但在第一次遇到类时添加标题
                docs.append(f"### Module: `{rel_path}`\n")
                module_doc = "placeholder" # 防止重复添加标题

            class_doc = ast.get_docstring(node) or "*No docstring*"
            docs.append(f"#### Class `{node.name}`")
            docs.append(f"```python\nclass {node.name}\n```")
            docs.append(f"{class_doc}\n")

            # 扫描类方法
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name.startswith('_') and item.name != '__init__':
                        continue  # 跳过私有方法
                    method_doc = ast.get_docstring(item) or "*No docstring*"
                    args = [a.arg for a in item.args.args]
                    if 'self' in args: args.remove('self')
                    args_str = ", ".join(args)
                    docs.append(f"- **Method** `{item.name}({args_str})`")
                    docs.append(f"  > {method_doc.splitlines()[0] if method_doc else ''}\n")

        elif isinstance(node, ast.FunctionDef):
            if node.name.startswith('_'):
                continue # 跳过私有函数

            has_content = True
            if not module_doc:
                docs.append(f"### Module: `{rel_path}`\n")
                module_doc = "placeholder"

            func_doc = ast.get_docstring(node) or "*No docstring*"
            args = [a.arg for a in node.args.args]
            args_str = ", ".join(args)

            docs.append(f"#### Function `{node.name}`")
            docs.append(f"```python\ndef {node.name}({args_str})\n```")
            docs.append(f"{func_doc}\n")

    return docs if has_content else []

def generate_api_docs(source_dir: str, output_file: str):
    """遍历目录并生成 Markdown 文档"""
    content = ["# Algo QOL API Reference\n", "Automated generated documentation.\n"]

    for root, dirs, files in os.walk(source_dir):
        # 排序以保持输出稳定
        dirs.sort()
        files.sort()

        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, source_dir)

                # 排除 migrations, tests, setup.py 等干扰项
                if 'migrations' in rel_path or 'tests' in rel_path or 'setup.py' in rel_path:
                    continue

                file_docs = parse_file(full_path, rel_path)
                content.extend(file_docs)
                if file_docs:
                    content.append("---\n")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(content))
    print(f"Documentation generated at {output_file}")

if __name__ == "__main__":
    # 假设脚本在 tools/ 下，项目根目录在上一级
    # 或者直接指定绝对路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    target_dir = os.path.join(project_root, 'algo_qol')
    output_md = os.path.join(project_root, 'docs', 'API_REFERENCE.md')

    # 确保扫描的是 algo_qol 目录
    if not os.path.exists(target_dir):
        # Fallback: 如果脚本被放在根目录运行
        target_dir = os.path.join(os.path.dirname(__file__), 'algo_qol')
        output_md = os.path.join(os.path.dirname(__file__), 'docs', 'API_REFERENCE.md')

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_md), exist_ok=True)

    print(f"Scanning {target_dir}...")
    generate_api_docs(target_dir, output_md)
