# -*- coding: utf-8 -*-
"""
@Time ： 2025/4/17 15:59
@Auth ： heshuai.sec@gmail.com
@File ：vllm_client.py

ref: https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_client_for_multimodal.py
"""
import base64
import inspect
import io
import json
import logging
import math
import re
from typing import Any, Optional, List, Union
from typing import Dict

import numpy as np
import requests
from PIL import Image
from openai import OpenAI

# 配置日志
logger = logging.getLogger(__name__)


def extract_value_from_text(text: str, key: str) -> Optional[Any]:
    """
    使用正则表达式从文本中提取。

    Args:
        text (str): 包含JSON数据的字符串
        key (str): 要提取的键名

    Returns:
        Optional[Any]: 提取到的值，如果未找到则返回None
    """
    return _extract_from_malformed_json(text, key)


def _extract_from_malformed_json(text: str, key: str) -> Optional[Any]:
    """
    从格式不正确的JSON文本中提取指定key的value。

    Args:
        text (str): 格式不正确的JSON文本
        key (str): 要提取的键名

    Returns:
        Optional[Any]: 提取到的值
    """
    # 先尝试标准JSON解析
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and key in parsed:
            return parsed[key]
    except (json.JSONDecodeError, TypeError):
        pass

    # 使用更精确的正则表达式匹配，考虑嵌套结构
    value_str = _extract_value_string(text, key)
    if value_str is None:
        return None

    # 尝试解析不同类型的值
    return _parse_value(value_str)


def _extract_value_string(text: str, key: str) -> Optional[str]:
    """
    提取指定key对应的value字符串，处理嵌套结构。

    Args:
        text (str): 包含JSON数据的文本
        key (str): 要提取的键名

    Returns:
        Optional[str]: 提取到的value字符串
    """
    # 查找key的位置
    key_pattern = rf'"{re.escape(key)}"\s*:'
    match = re.search(key_pattern, text)
    if not match:
        return None

    # 从冒号后开始提取值
    start_pos = match.end()
    value_start = start_pos

    # 跳过空白字符
    while value_start < len(text) and text[value_start].isspace():
        value_start += 1

    if value_start >= len(text):
        return None

    # 根据第一个字符确定值的类型并提取
    first_char = text[value_start]

    if first_char == '"':
        # 字符串值
        return _extract_string_value(text, value_start)
    elif first_char == '[':
        # 数组值
        return _extract_bracket_value(text, value_start, '[', ']')
    elif first_char == '{':
        # 对象值
        return _extract_bracket_value(text, value_start, '{', '}')
    else:
        # 基本类型值（数字、布尔值、null）
        return _extract_primitive_value(text, value_start)


def _extract_string_value(text: str, start_pos: int) -> str:
    """提取字符串值，处理转义字符。"""
    if text[start_pos] != '"':
        return ""

    pos = start_pos + 1
    result = '"'

    while pos < len(text):
        char = text[pos]
        result += char

        if char == '"' and text[pos - 1] != '\\':
            break
        elif char == '\\' and pos + 1 < len(text):
            # 处理转义字符
            pos += 1
            if pos < len(text):
                result += text[pos]

        pos += 1

    return result


def _extract_bracket_value(text: str, start_pos: int, open_bracket: str, close_bracket: str) -> str:
    """提取括号包围的值（数组或对象）。"""
    if text[start_pos] != open_bracket:
        return ""

    bracket_count = 0
    pos = start_pos
    result = ""
    in_string = False
    escape_next = False

    while pos < len(text):
        char = text[pos]
        result += char

        if escape_next:
            escape_next = False
        elif char == '\\':
            escape_next = True
        elif char == '"' and not escape_next:
            in_string = not in_string
        elif not in_string:
            if char == open_bracket:
                bracket_count += 1
            elif char == close_bracket:
                bracket_count -= 1
                if bracket_count == 0:
                    break

        pos += 1

    return result


def _extract_primitive_value(text: str, start_pos: int) -> str:
    """提取基本类型值（数字、布尔值、null）。"""
    pos = start_pos
    result = ""

    while pos < len(text):
        char = text[pos]

        # 遇到分隔符或结束符时停止
        if char in [',', '}', ']', '\n', '\r'] or (char.isspace() and result):
            break

        if not char.isspace() or not result:
            result += char

        pos += 1

    return result.strip()


def _parse_value(value_str: str) -> Any:
    """
    解析值字符串，返回对应的Python对象。

    Args:
        value_str (str): 值的字符串表示

    Returns:
        Any: 解析后的Python对象
    """
    value_str = value_str.strip()

    # 处理数组
    if value_str.startswith('[') and value_str.endswith(']'):
        return _parse_array(value_str)

    # 处理对象
    if value_str.startswith('{') and value_str.endswith('}'):
        try:
            return json.loads(value_str)
        except:
            return value_str

    # 处理字符串
    if value_str.startswith('"') and value_str.endswith('"'):
        return value_str[1:-1]  # 去除引号

    # 处理布尔值
    if value_str.lower() == 'true':
        return True
    elif value_str.lower() == 'false':
        return False

    # 处理null
    if value_str.lower() == 'null':
        return None

    # 处理数字
    try:
        if '.' in value_str:
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        pass

    # 如果都不匹配，返回原字符串
    return value_str


def _parse_array(array_str: str) -> List[Any]:
    """
    解析数组字符串。

    Args:
        array_str (str): 数组的字符串表示，如 "[1, 2, 3]"

    Returns:
        List[Any]: 解析后的列表
    """
    try:
        # 尝试直接JSON解析
        return json.loads(array_str)
    except:
        # JSON解析失败，手动解析
        content = array_str[1:-1].strip()  # 去除方括号
        if not content:
            return []

        # 简单的分割处理，支持基本的数组格式
        elements = []
        current_element = ""
        bracket_count = 0
        in_quotes = False
        quote_char = None

        for char in content:
            if char in ['"', "'"] and not in_quotes:
                in_quotes = True
                quote_char = char
                current_element += char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                current_element += char
            elif char in ['{', '['] and not in_quotes:
                bracket_count += 1
                current_element += char
            elif char in ['}', ']'] and not in_quotes:
                bracket_count -= 1
                current_element += char
            elif char == ',' and bracket_count == 0 and not in_quotes:
                elements.append(_parse_value(current_element.strip()))
                current_element = ""
            else:
                current_element += char

        # 添加最后一个元素
        if current_element.strip():
            elements.append(_parse_value(current_element.strip()))

        return elements


def parse_llm_json(text: str) -> Optional[dict]:
    """
    Parse JSON output from LLM with mild repair on common formatting issues.

    Args:
        text (str): LLM输出的文本，包含JSON数据

    Returns:
        Optional[dict]: 解析后的JSON对象，失败时返回None
    """

    def _mild_repair(json_str: str) -> str:
        """
        Minimal, *safe* fixes for common LLM slip-ups:
          • tuples (…)  → arrays […]
          • single quotes → double quotes
          • unescaped \n/\t inside strings → escaped
          • trailing commas before } or ] → removed
        """
        # (…) → […]
        json_str = re.sub(
            r"\(\s*([0-9\-\s.,]+?)\s*\)",
            lambda m: "[" + m.group(1) + "]",
            json_str,
        )

        # protect control characters inside quoted strings
        def _escape_ctrl(match: re.Match) -> str:
            return match.group(0).replace("\n", "\\n").replace("\t", "\\t")

        json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', _escape_ctrl, json_str, flags=re.S)

        # single-quote strings → double-quote
        json_str = re.sub(r"'", '"', json_str)

        # Remove trailing commas before } or ]
        json_str = re.sub(r",\s*(\}|\])", r"\1", json_str)

        return json_str

    # Extract JSON block by matching outermost braces
    depth = 0
    start = end = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if start is None or end is None:
        return None

    json_text = text[start:end]
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        try:
            repaired = _mild_repair(json_text)
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            logger.error(f'Failed to parse JSON: {e}')
            logger.error(f'Original text: {text}')
            return None


def parse_cot_response(response: str, parse_answer: bool = True) -> dict:
    """
    Parses a CoT (Chain of Thought) response string, tries to fix malformed tags or JSON.

    Args:
        response (str): LLM-generated CoT response string.
        parse_answer (bool): 是否尝试将answer部分解析为JSON，默认True

    Returns:
        dict: {
            'think': str or None,  # 思考过程
            'answer': dict/str or None  # 答案内容，解析成功时为dict，否则为原始字符串
        }
    """
    # Try to extract <think> section
    try:
        think_match = re.search(r"<think>\s*(.*?)\s*</think>", response, re.DOTALL)
        think_text = think_match.group(1).strip() if think_match else ''
    except:
        think_text = ''

    # Try to extract <answer> block (first {} block after <answer>)
    try:
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
        answer_text = answer_match.group(1).strip() if answer_match else response.strip()
    except:
        answer_text = response.strip()

    # Try to parse answer JSON
    answer_data = None
    if answer_text and parse_answer:
        answer_data = parse_llm_json(answer_text)

    if answer_data is None:
        answer_data = answer_text

    return {
        "think": think_text,
        "answer": answer_data
    }


def extract_json_blocks(text: str) -> List[dict]:
    """
    从文本中提取所有JSON块。

    Args:
        text (str): 包含JSON数据的文本

    Returns:
        List[dict]: 提取到的JSON对象列表
    """
    json_blocks = []
    i = 0

    while i < len(text):
        if text[i] == '{':
            # 找到JSON块的开始
            depth = 0
            start = i

            while i < len(text):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        # 找到完整的JSON块
                        json_text = text[start:i + 1]
                        try:
                            parsed = json.loads(json_text)
                            json_blocks.append(parsed)
                        except json.JSONDecodeError:
                            # 尝试修复后再解析
                            parsed = parse_llm_json(json_text)
                            if parsed:
                                json_blocks.append(parsed)
                        break
                i += 1
        else:
            i += 1

    return json_blocks


def extract_code_blocks(text: str, language: str = None) -> List[str]:
    """
    从文本中提取代码块。

    Args:
        text (str): 包含代码块的文本
        language (str, optional): 指定语言类型，如'python', 'json'等

    Returns:
        List[str]: 提取到的代码块列表
    """
    if language:
        # 匹配特定语言的代码块
        pattern = rf'```{re.escape(language)}\s*\n(.*?)```'
    else:
        # 匹配所有代码块
        pattern = r'```(?:\w+)?\s*\n(.*?)```'

    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def clean_llm_output(text: str) -> str:
    """
    清理LLM输出中的常见问题。

    Args:
        text (str): 原始LLM输出

    Returns:
        str: 清理后的文本
    """
    # 移除多余的空白字符
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    # 移除开头和结尾的空白字符
    text = text.strip()

    # 修复常见的标点符号问题
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)

    return text


def extract_structured_response(text: str,
                                sections: List[str] = None,
                                default_sections: List[str] = None) -> dict:
    """
    从结构化LLM响应中提取各个部分。

    Args:
        text (str): LLM响应文本
        sections (List[str], optional): 要提取的部分名称列表
        default_sections (List[str], optional): 默认部分列表

    Returns:
        dict: 提取的各部分内容
    """
    if sections is None:
        sections = default_sections or ['think', 'answer', 'explanation', 'reasoning']

    result = {}

    for section in sections:
        # 尝试多种标签格式
        patterns = [
            rf'<{section}>\s*(.*?)\s*</{section}>',  # XML风格
            rf'#{section.upper()}\s*\n(.*?)(?=\n#|\Z)',  # Markdown风格
            rf'**{section.capitalize()}:**\s*\n(.*?)(?=\n\*\*|\Z)',  # 加粗风格
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                result[section] = content
                break

        # 如果没有找到，设置为None
        if section not in result:
            result[section] = None

    return result


def get_bool_value_score(response: Dict, label_key: str = 'result', *args, **kwargs) -> tuple:
    """
    return None, None if fail
    """
    try:
        label = extract_value_from_text(response['choices'][0]['message']['content'], label_key)
        if label is None:
            return None, None
        label = int(label)
        # 4. 从logprobs中提取分数
        logprobs_data = response['choices'][0].get('logprobs')
        # always return score 1. if no logprobs
        if not logprobs_data or 'content' not in logprobs_data:
            return label, 1.0

        # 5. 在token序列中查找true/false对应的logprob
        tokens = logprobs_data['content']

        # Find the token corresponding to the result
        target_idx = -1
        key_found = False
        label_key_lower = label_key.lower()

        # First pass: try to find key then value
        for i, token_info in enumerate(tokens):
            token = token_info.get('token', '').strip().lower()
            clean_token = token.replace('"', '').replace("'", '').replace(' ', '').replace(',', '').replace(':', '')

            if not key_found:
                if label_key_lower in clean_token:
                    key_found = True
            else:
                if 'true' in clean_token or 'false' in clean_token or 'yes' in clean_token or 'no' in clean_token:
                    target_idx = i
                    break

        # Fallback: find first boolean if key not found
        if target_idx == -1:
            for i, token_info in enumerate(tokens):
                token = token_info.get('token', '').strip().lower()
                if 'true' in token or 'false' in token or 'yes' in token or 'no' in token:
                    target_idx = i
                    break

        score = 1.0
        if target_idx != -1:
            token_info = tokens[target_idx]
            top_logprobs = token_info.get('top_logprobs', [])

            prob_true = 0.0
            prob_false = 0.0

            if top_logprobs:
                for top_item in top_logprobs:
                    tok = top_item.get('token', '').strip().lower()
                    clean_tok = tok.replace('"', '').replace("'", '').replace(' ', '').replace(',', '').replace(':', '')
                    lp = top_item.get('logprob', -100.0)
                    p = math.exp(lp)
                    if clean_tok in ['true', 'yes']:
                        prob_true += p
                    elif clean_tok in ['false', 'no']:
                        prob_false += p
            else:
                tok = token_info.get('token', '').strip().lower()
                clean_tok = tok.replace('"', '').replace("'", '').replace(' ', '').replace(',', '').replace(':', '')
                lp = token_info.get('logprob', 0.0)
                p = math.exp(lp)
                if clean_tok in ['true', 'yes']:
                    prob_true = p
                elif clean_tok in ['false', 'no']:
                    prob_false = p

            total_prob = prob_true + prob_false
            if total_prob > 0:
                if label == 1:
                    score = prob_true / total_prob
                else:
                    score = prob_false / total_prob
            else:
                # Fallback
                score = math.exp(token_info.get('logprob', 0.0))

        return label, score

    except Exception as e:
        logger.error(f"Error in get_l1_label_score: {e}")
        return None, None


def get_int_value_score(response: Dict, label_key='confidence', score_method='weighted_avg', max_range=5, *args,
                        **kwargs) -> tuple:
    """
    score_method: weighted_avg, normalized_avg, top1
    """

    def _get_confidence_result(content):
        """
        extract confidence score from content, return None if fail
        Args:
            content: response content string

        Returns:
            confidence: confidence label, return None if fail
        """
        try:
            answer = parse_cot_response(content)['answer']
            confidence_label = None
            if isinstance(answer, dict):
                confidence_label = answer.get(label_key, None)
            if confidence_label is None:
                confidence_label = extract_value_from_text(content, label_key)
                if confidence_label is not None:
                    confidence_label = int(confidence_label)
            return confidence_label
        except Exception:
            return None

    def _calculate_score_from_logprobs(top_logprobs, method):
        """
        Calculate score from logprobs based on different methods
        """
        token2score = {i: 0.0 for i in range(max_range + 1)}
        for token_info in top_logprobs:
            logprob = token_info.get('logprob', 0.0)
            prob = math.exp(logprob)
            clean_token = token_info.get('token', '').strip().replace('"', '').replace("'", '').replace(' ',
                                                                                                        '').replace(',',
                                                                                                                    '').replace(
                ':', '')
            if clean_token.isdigit() and 0 <= int(clean_token) <= max_range:
                token2score[int(clean_token)] += prob
        scores = np.array(list(token2score.values()))
        if np.sum(scores) == 0:
            return 1.0
        if method == 'normalized_avg':
            scores = scores / np.sum(scores)
            total = 0.
            for i in range(max_range + 1):
                total += i * scores[i]
            return total / max_range
        elif method == 'weighted_avg':
            total = 0.
            for i in range(max_range + 1):
                total += i * scores[i]
            return total / max_range
        else:
            NotImplementedError(f"method {method} not supported")

    def _get_score_from_response(response, method):
        """
        Extract confidence score from VLLM response logprobs
        """
        logprobs_data = response['choices'][0].get('logprobs')
        if not logprobs_data or 'content' not in logprobs_data:
            return 1.0

        tokens = logprobs_data['content']

        # 查找confidence关键字的位置
        confidence_idx = -1
        for i, token_info in enumerate(tokens):
            token = token_info.get('token', '').strip()
            # 清理token并检查是否包含confidence关键字
            clean_token = token.replace('"', '').replace("'", '').replace(' ', '').replace(',', '').replace(':', '')
            if label_key.lower() in clean_token.lower():
                confidence_idx = i
                break

        # 如果找到了confidence，从其后面开始查找第一个数字token
        if confidence_idx != -1:
            for i in range(confidence_idx + 1, len(tokens)):
                token = tokens[i].get('token', '').strip()
                # 清理token（去除引号、空格、逗号等）
                clean_token = token.replace('"', '').replace("'", '').replace(' ', '').replace(',', '').replace(':', '')

                # 检查是否是数字token
                if clean_token.isdigit():
                    # 获取该token的top_logprobs
                    top_logprobs = tokens[i].get('top_logprobs', [])
                    if method == 'top1' or top_logprobs is None:
                        logprob = tokens[i].get('logprob', -100.)
                        return math.exp(logprob)
                    else:
                        return _calculate_score_from_logprobs(top_logprobs, method)
                    break
        return 1.0

    try:
        fail_result = (None, None)

        content = response['choices'][0]['message']['content']
        confidence = _get_confidence_result(content)
        if confidence is None:
            return fail_result

        score = _get_score_from_response(response, score_method)
        return confidence, score

    except Exception as e:
        logger.error(f"Error in get_int_value_score: {e}")
        return fail_result


def get_list_value_score(response: Dict, label_key: str = 'labels') -> tuple:
    def _get_list_value(content):
        # TODO: try to extract list of index from string if fail to parse the json
        parsed = parse_cot_response(content)
        answer = parsed['answer']
        l2_labels = None
        if isinstance(answer, dict):
            l2_labels = answer.get(label_key, None)
        if l2_labels is None:
            l2_labels = extract_value_from_text(answer, label_key)
        return l2_labels

    fail_result = None, None
    try:
        list_value = _get_list_value(response['choices'][0]['message']['content'])
        if list_value is None:
            raise ValueError('cant find labels')
        if list_value is not None and len(list_value) == 0:
            return [], []

        logprobs_data = response['choices'][0].get('logprobs')
        if logprobs_data is None:
            return list_value, [1.] * len(list_value)
        tokens = logprobs_data['content']
        # 查找 labels 或 labels 关键字在 token 序列中的位置
        key_found_idx = -1
        for i, token_info in enumerate(tokens):
            token = token_info.get('token', '').strip()
            if len(token) and token in label_key:
                key_found_idx = i
                break

        # 如果找不到关键字，返回默认分数
        if key_found_idx == -1:
            return fail_result

        # 收集 labels/labels 之后在 [] 内的数字token及其概率
        digit_tokens = []
        in_array = False
        for i in range(key_found_idx + 1, len(tokens)):
            token = tokens[i].get('token', '').strip()
            # 检测数组开始
            if '[' in token:
                in_array = True
                continue

            # 检测数组结束
            if ']' in token:
                break

            # 在数组内收集数字token
            if in_array:
                # 清理token（去除引号和空格）
                clean_token = token.replace('"', '').replace("'", '').replace(' ', '').replace(',', '')

                # 检查是否是数字
                if clean_token.isdigit():
                    logprob = tokens[i].get('logprob', 0.0)
                    prob = math.exp(logprob)
                    digit_tokens.append((clean_token, prob))
        hit_labels = [item[0] for item in digit_tokens]
        label_scores = [item[1] for item in digit_tokens]
        return hit_labels, label_scores

    except Exception as e:
        logger.error(f"Error in get_list_value_score: {e}")
        return fail_result


def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""
    with requests.get(content_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode('utf-8')
    return result


def encode_image(image: Union[str, np.ndarray, Image.Image]):
    """
    encoder local image to base64 string, either from local file path or numpy array or PIL Image.
    Args:
        image:

    Returns:

    """
    _valid_image_format = ['.jpg', '.png', '.jpeg', '.bmp', '.rgb', '.tif', '.tiff', '.webp', '.gif', 'jfif']

    def _check_image_file(path):
        return any([path.lower().endswith(e) for e in _valid_image_format])

    # image path
    if isinstance(image, str):
        if _check_image_file(image):
            with open(image, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        else:
            return image  # already base64
    # numpy array
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    # PIL Image
    if isinstance(image, Image.Image):
        buff = io.BytesIO()
        image.save(buff, format="JPEG")

        # Encode bytes to base64 string
        img_base64 = base64.b64encode(buff.getvalue()).decode('utf-8')
        return img_base64
    else:
        return None


def make_message(prompt: str,
                 images: list = None,
                 videos: list = None,
                 system_prompt: str = None,
                 add_think_tag: bool = False,
                 *args, **kwargs
                 ):
    """
    make a typical sandwich(system_prompt-mm_data(images/videos)-txt) messages for MLLM completion.
    Args:
        prompt:
        images:
        system_prompt:
    Returns:
    """
    SKIP_THINK_CONTENT = {"role": "assistant", "content": f'<think></think>'}

    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    msg = {"role": "user", "content": []}
    if images is not None:
        if not isinstance(images, list):
            images = [images]
        for image in images:
            if isinstance(image, str) and (image.startswith("http://") or image.startswith("https://")):
                msg["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": image
                    },
                })
            else:
                base64_image = encode_image(image)
                if base64_image is None: raise Exception("invalid image")
                msg["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                })
    if videos is not None:
        if not isinstance(videos, list):
            videos = [videos]
        # TODO: check this implementation
        for video in videos:
            if isinstance(video, str) and (video.startswith("http://") or video.startswith("https://")):
                msg["content"].append({
                    "type": "video_url",
                    "video_url": {
                        "url": video
                    },
                })
            else:
                base64_video = encode_base64_content_from_url(video)
                msg["content"].append({
                    "type": "video_url",
                    "video_url": {
                        "url": f"data:video/mp4;base64,{base64_video}"
                    },
                })
    msg["content"].append({
        "type": "text",
        "text": prompt
    })
    messages.append(msg)
    if add_think_tag:
        messages.append(SKIP_THINK_CONTENT)
    return messages


def get_content(response):
    content = [item.message.content for item in response.choices]
    if len(content) == 1:
        content = content[0]
    return content


class VLLMClient:
    def __init__(self, api_base, api_key="EMPTY", model=None, system_prompt=None, *args, **kwargs):
        self.openai_api_key = api_key
        self.openai_api_base = api_base
        self.client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_api_base,
        )
        self.model = model
        self.system_prompt = system_prompt
        self.model_list = [item.id for item in self.client.models.list().data]
        if self.model is None:
            logger.info('all available models: {}'.format(self.model_list))
            self.model = self.model_list[0]
            logger.info('use default model: {}'.format(self.model))

    def set_model(self, model: str):
        if model not in self.model_list:
            return False
        self.model = model
        return True

    def __call__(self, prompt: str = None,
                 images: Optional[Union[str, list]] = None,
                 videos: Optional[Union[str, list]] = None,  # Todo: add and evaluate video support
                 messages: Optional[list] = None,  # will override prompt, images, videos if not None
                 no_think: Optional[bool] = False,  # only valid for thinking model
                 return_raw_response=False,  # return raw response from openai api
                 system_prompt: Optional[str] = None,  # will override self.system_prompt if not None
                 model: str = None,
                 *args, **kwargs):
        if model is None:
            model = self.model
        if system_prompt is None:
            system_prompt = self.system_prompt
        if messages is None:
            messages = make_message(prompt, images, system_prompt=system_prompt, add_think_tag=no_think)
        # get params of self.client.chat.completions.create. remove kwargs that not in the function signature

        params = {k: v for k, v in kwargs.items() if
                  k in inspect.signature(self.client.chat.completions.create).parameters}
        chat_completion_from_url = self.client.chat.completions.create(
            messages=messages,
            model=model,
            **params,
        )
        if return_raw_response:
            return chat_completion_from_url
        return chat_completion_from_url.choices[0].message.content


if __name__ == '__main__':
    # test
    client = VLLMClient(api_base='http://7.129.26.12:8102/v1')

    prompt = '请仔细观察和分析图像，生成一段caption'
    img_url = 'https://fastly.picsum.photos/id/0/5000/3333.jpg?hmac=_j6ghY5fCfSD6tvtcV74zXivkJSPIfR9B8w34XeQmvU'
    # load image to Image
    img = Image.open(requests.get(img_url, stream=True).raw)
    img = img.convert('RGB').resize((720, 720))
    message = make_message(prompt, images=[img])
    print(message)
    response = client(messages=message, max_completion_tokens=4096)
    print(response)
