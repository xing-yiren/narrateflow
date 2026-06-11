"""Unit tests for utility modules"""
import sys
sys.path.insert(0, '.')

from narrateflow.utils.schema_validator import extract_json_from_text, validate_vlm_output
from narrateflow.utils.ffprobe import _parse_fraction

def test_extract_json_plain():
    result = extract_json_from_text('{"key": "value"}')
    assert result == {"key": "value"}, f"Got {result}"

def test_extract_json_with_thinking():
    result = extract_json_from_text('<thinking>thought</thinking>\n{"a": 1}')
    assert result == {"a": 1}, f"Got {result}"

def test_extract_json_markdown():
    result = extract_json_from_text('```json\n{"b": 2}\n```')
    assert result == {"b": 2}, f"Got {result}"

def test_extract_json_trailing_comma():
    result = extract_json_from_text('{"c": 3,}')
    assert result == {"c": 3}, f"Got {result}"

def test_validate_vlm_output_ok():
    ok, err = validate_vlm_output({"visual_summary": "test", "physical_tags": ["a"]})
    assert ok, f"Expected OK, got: {err}"

def test_validate_vlm_output_missing():
    ok, err = validate_vlm_output({"physical_tags": ["a"]})
    assert not ok

def test_parse_fraction():
    assert abs(_parse_fraction("30000/1001") - 29.97) < 0.01
    assert _parse_fraction("30") == 30.0
    assert _parse_fraction("0/0") == 0.0

def test_stage0_text_processing():
    """Test Stage 0 text processing independently"""
    from narrateflow.stage0_manifest import (
        number_to_chinese, 
        split_narration, 
        _count_chinese,
        apply_pronunciation_dict
    )
    
    # 数字转中文
    assert number_to_chinese("第3章") == "第三章"
    assert number_to_chinese("2024年") == "二千零二十四年"
    
    # 分句
    text = "今天天气很好。我们一起去公园吧！那里有很多花。"
    segments = split_narration(text, min_chars=1, max_chars=30)
    assert len(segments) >= 1, f"Got {len(segments)} segments"
    
    # 中文字符统计
    assert _count_chinese("Hello世界123") == 2
    
    # 发音词典
    p_dict = {"AI": "人工智能", "API": "接口"}
    result = apply_pronunciation_dict("使用AI和API", p_dict)
    assert "人工智能" in result
    assert "接口" in result
    
    print("✓ All Stage 0 text processing tests passed")

if __name__ == "__main__":
    test_extract_json_plain()
    test_extract_json_with_thinking()
    test_extract_json_markdown()
    test_extract_json_trailing_comma()
    test_validate_vlm_output_ok()
    test_validate_vlm_output_missing()
    test_parse_fraction()
    test_stage0_text_processing()
    print("\n✓ All util tests passed!")
