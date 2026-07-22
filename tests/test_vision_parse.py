import pytest

from app.services.vision import VisionService

FULL = (
    '{"subj":"cat","sec":["basket"],"obj":["cat","basket","window"],'
    '"scene":"living room","env":"indoor","sty":"realistic","med":"photo",'
    '"mat":["fabric"],"lit":"natural light","mood":"calm","persp":"eye level",'
    '"tex":"soft","pat":"plain"}'
)


class TestParseResponse:
    """Parsing/extraction logic in isolation — no model needed."""

    def test_clean_json_is_mapped_to_pipeline_fields(self):
        result = VisionService._parse_response(FULL)
        assert result["primary_subject"] == "cat"
        assert result["scene"] == "living room"
        assert result["style"] == "realistic"
        assert result["art_medium"] == "photo"
        assert result["objects"] == ["cat", "basket", "window"]
        assert result["materials"] == ["fabric"]

    def test_json_with_markdown_fences(self):
        result = VisionService._parse_response(f"```json\n{FULL}\n```")
        assert result["primary_subject"] == "cat"

    def test_json_with_surrounding_text(self):
        result = VisionService._parse_response(f"Here you go:\n{FULL}\nDone.")
        assert result["primary_subject"] == "cat"

    def test_missing_keys_are_absent_not_unknown(self):
        result = VisionService._parse_response('{"subj":"car","sty":"minimalist"}')
        assert result["primary_subject"] == "car"
        # Downstream treats absence as "did not observe"; a literal "Unknown"
        # would leak into titles and traits.
        assert "mood" not in result
        assert "lighting" not in result

    def test_comma_separated_string_becomes_list(self):
        result = VisionService._parse_response('{"obj":"sofa, lamp and rug"}')
        assert result["objects"] == ["sofa", "lamp", "rug"]

    def test_list_value_for_scalar_field_is_flattened(self):
        result = VisionService._parse_response('{"mood":["calm","quiet"]}')
        assert result["mood"] == "calm"

    def test_truncated_output_is_repaired(self):
        # The model ran out of tokens mid-object; salvage what arrived.
        truncated = '{"subj":"dragon","scene":"mountain","obj":["wings","sca'
        result = VisionService._parse_response(truncated)
        assert result["primary_subject"] == "dragon"
        assert result["scene"] == "mountain"

    def test_unknown_keys_are_ignored(self):
        result = VisionService._parse_response('{"subj":"tree","bogus":"junk"}')
        assert result == {"primary_subject": "tree"}

    def test_empty_strings_are_kept_out_of_lists(self):
        result = VisionService._parse_response('{"obj":["sofa","",  "  "]}')
        assert result["objects"] == ["sofa"]

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            VisionService._parse_response("This is just text with no JSON")

    def test_non_object_json_raises(self):
        with pytest.raises(ValueError):
            VisionService._parse_response("[1, 2, 3]")
