"""
JSON Schema 校验 + 字段范围检查
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# 加载内置 schemas
_SCHEMA_CACHE: Dict[str, Dict] = {}


def _load_schema(schema_name: str) -> Dict:
    """加载 JSON Schema 文件并缓存"""
    if schema_name in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[schema_name]
    
    schema_dir = Path(__file__).parent.parent / "schemas"
    schema_path = schema_dir / f"{schema_name}.json"
    
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema 文件不存在: {schema_path}")
    
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    
    _SCHEMA_CACHE[schema_name] = schema
    return schema


def validate_json(data: Dict, schema_name: str) -> Tuple[bool, Optional[str]]:
    """
    用 JSON Schema 校验数据。
    
    Returns:
        (is_valid, error_message)
    """
    try:
        import jsonschema
    except ImportError:
        logger.warning("jsonschema 未安装，跳过 Schema 校验")
        return True, None
    
    schema = _load_schema(schema_name)
    
    try:
        jsonschema.validate(instance=data, schema=schema)
        return True, None
    except jsonschema.ValidationError as e:
        error_msg = f"Schema 校验失败 [{schema_name}]: {e.message}"
        logger.warning(error_msg)
        return False, error_msg
    except jsonschema.SchemaError as e:
        error_msg = f"Schema 定义错误 [{schema_name}]: {e.message}"
        logger.error(error_msg)
        return False, error_msg


def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    从 LLM 输出中提取 JSON 对象。
    支持场景：
    1. 纯 JSON 文本
    2. markdown 代码块包裹 ```json ... ```
    3. 去 <thinking> 块后提取
    4. 正则兜底：找最外层 {} 
    """
    import re
    
    # 1. 去除 <thinking>...</thinking> 块
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    text = re.sub(r'<Think>.*?</Think>', '', text, flags=re.DOTALL)
    
    text = text.strip()
    
    # 2. 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 3. 去除 markdown 代码块标记
    md_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # 4. 正则兜底：找最外层 { }，也处理截断 JSON（缺少 }）
    brace_match = re.search(r'\{.*\}', text, re.DOTALL)
    truncated_match = None
    if not brace_match:
        # 找开头 { 但没有结尾 } 的截断 JSON
        truncated_match = re.search(r'\{.*$', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # 5. 尝试修复常见 LLM 输出错误（尾部逗号等）
    if brace_match:
        candidate = brace_match.group(0)
        # 移除尾部逗号
        candidate = re.sub(r',(\s*[}\]])', r'\1', candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 6. Thinking 模型截断修复：自动闭合未完成的括号
    repair_match = brace_match or truncated_match
    if repair_match:
        candidate = repair_match.group(0).rstrip()
        # 计算未闭合的括号
        open_braces = candidate.count("{") - candidate.count("}")
        open_brackets = candidate.count("[") - candidate.count("]")
        # 检查是否在字符串内部被截断（最后一个 " 是否闭合）
        if candidate.count('"') % 2 != 0:
            candidate += '"'  # 闭合最后一个字符串
        # 先闭合内层括号（数组），再闭合外层（对象）
        candidate += "]" * max(0, open_brackets) + "}" * max(0, open_braces)
        # 再次移除尾部逗号
        candidate = re.sub(r',(\s*[}\]])', r'\1', candidate)
        try:
            result = json.loads(candidate)
            logger.warning(
                "JSON truncated but auto-repaired "
                "(+%d braces, +%d brackets)", open_braces, open_brackets
            )
            return result
        except json.JSONDecodeError:
            pass

    logger.error(f"无法从文本中提取 JSON: {text[:200]}...")
    return None


def validate_vlm_output(data: Dict) -> Tuple[bool, Optional[str]]:
    """
    VLM 输出专项校验：
    - confidence ∈ [0, 1]
    - 必填字段非空
    """
    # 检查 vlm_output 结构
    if "visual_summary" not in data:
        return False, "缺少必填字段: visual_summary"
    if "physical_tags" not in data:
        return False, "缺少必填字段: physical_tags"
    
    if not data.get("visual_summary"):
        return False, "visual_summary 为空"
    if not isinstance(data.get("physical_tags"), list):
        return False, "physical_tags 不是数组"
    
    # 检查 action_candidates 的 confidence
    for action in data.get("action_candidates", []):
        conf = action.get("confidence")
        if conf is not None and not (0 <= conf <= 1):
            return False, f"action confidence 超出 [0,1]: {conf}"
    
    return True, None
