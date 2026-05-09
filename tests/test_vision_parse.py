import pytest

from app.services.vision import VisionService


class TestParseResponse:
    """Test the JSON parsing/extraction logic in isolation (no model needed)."""

    def test_clean_json(self):
        raw = '{"object_type":"cat","style":"Realistic","dominant_color":"Orange","mood":"Calm","lighting":"Natural","environment":"Garden"}'
        result = VisionService._parse_response(raw)
        assert result["object_type"] == "cat"
        assert result["style"] == "Realistic"

    def test_json_with_markdown_fences(self):
        raw = '```json\n{"object_type":"dog","style":"Abstract","dominant_color":"Blue","mood":"Joyful","lighting":"Soft","environment":"Studio"}\n```'
        result = VisionService._parse_response(raw)
        assert result["object_type"] == "dog"

    def test_json_with_surrounding_text(self):
        raw = 'Here is the analysis:\n{"object_type":"tree","style":"Watercolor","dominant_color":"Green","mood":"Serene","lighting":"Natural","environment":"Forest"}\nDone.'
        result = VisionService._parse_response(raw)
        assert result["object_type"] == "tree"

    def test_missing_keys_filled_with_unknown(self):
        raw = '{"object_type":"car","style":"Minimalist"}'
        result = VisionService._parse_response(raw)
        assert result["dominant_color"] == "Unknown"
        assert result["mood"] == "Unknown"
        assert result["lighting"] == "Unknown"
        assert result["environment"] == "Unknown"

    def test_empty_values_become_unknown(self):
        raw = '{"object_type":"","style":"  ","dominant_color":"Blue","mood":"Dark","lighting":"Neon","environment":"City"}'
        result = VisionService._parse_response(raw)
        assert result["object_type"] == "Unknown"
        assert result["style"] == "Unknown"

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            VisionService._parse_response("This is just text with no JSON")

    def test_invalid_json_raises(self):
        with pytest.raises((ValueError, Exception)):
            VisionService._parse_response("{broken json")
