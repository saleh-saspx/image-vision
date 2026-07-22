from app.services.imagestats import ImageStats
from app.services.nft_generator import (
    generate_attributes,
    generate_description,
    generate_metadata,
    generate_tags,
    generate_title,
    infer_category,
    normalize_fields,
)

# Raw model output, as vision.VisionService.analyze would return it.
RAW = {
    "primary_subject": "a sofa set",
    "secondary_subjects": ["coffee table", "plant"],
    "objects": ["Sofa", "Armchair", "Coffee Table", "Plant", "Cup"],
    "scene": "living room",
    "environment": "inside",
    "style": "classical",
    "art_medium": "3d rendering",
    "materials": ["wooden", "upholstery"],
    "lighting": "natural light",
    "mood": "luxurious",
    "perspective": "frontal",
    "texture": "plush",
    "pattern": "solid",
}

STATS = ImageStats(
    colors=["Light Blue", "Brown", "White"],
    color_weights=[0.42, 0.31, 0.18],
    complexity="Medium",
    complexity_score=0.12,
    orientation="Landscape",
    aspect_ratio=1.5,
)


class TestNormalization:
    def test_synonyms_map_to_canonical_vocabulary(self):
        fields, _ = normalize_fields(RAW)
        assert fields["style"] == "Classic"
        assert fields["scene"] == "Living Room"
        assert fields["environment"] == "Indoor"
        assert fields["art_medium"] == "3D Render"
        assert fields["lighting"] == "Natural Daylight"
        assert fields["mood"] == "Luxury"
        assert fields["perspective"] == "Front View"
        assert fields["texture"] == "Soft"
        assert fields["pattern"] == "Plain"
        assert fields["materials"] == ["Wood", "Fabric"]

    def test_subject_articles_are_stripped(self):
        fields, _ = normalize_fields(RAW)
        assert fields["primary_subject"] == "Sofa Set"

    def test_unknown_values_are_dropped_not_stringified(self):
        fields, conf = normalize_fields({"style": "unknown", "mood": "", "scene": "n/a"})
        assert "style" not in fields
        assert "mood" not in fields
        assert "scene" not in fields
        assert conf == {}

    def test_recognised_terms_score_above_free_text(self):
        _, known_conf = normalize_fields({"style": "Cyberpunk"})
        _, unknown_conf = normalize_fields({"style": "wibbly wobbly"})
        assert known_conf["style"] > unknown_conf["style"]


class TestTitle:
    def test_title_is_descriptive_not_random(self):
        fields, _ = normalize_fields(RAW)
        fields["dominant_colors"] = STATS.colors
        title, _ = generate_title(fields)
        assert title == "Classic Light Blue Sofa Set"

    def test_title_has_no_collection_number(self):
        fields, _ = normalize_fields(RAW)
        title, _ = generate_title(fields)
        assert "#" not in title

    def test_title_does_not_repeat_words(self):
        fields = {"primary_subject": "Ceramic Vase", "materials": ["Ceramic"], "style": "Minimalist"}
        title, _ = generate_title(fields)
        assert title.lower().split().count("ceramic") == 1
        assert title == "Minimalist Ceramic Vase"

    def test_title_stays_short(self):
        fields, _ = normalize_fields(RAW)
        fields["dominant_colors"] = STATS.colors
        assert len(generate_title(fields)[0].split()) <= 5

    def test_title_survives_empty_input(self):
        title, conf = generate_title({})
        assert title
        assert conf > 0


class TestDescription:
    def test_description_mentions_key_facets(self):
        fields, _ = normalize_fields(RAW)
        fields["dominant_colors"] = STATS.colors
        desc, _ = generate_description(fields)
        lowered = desc.lower()
        assert "classic" in lowered       # style
        assert "sofa set" in lowered      # subject
        assert "living room" in lowered   # scene
        assert "light blue" in lowered    # colors
        assert "luxury" in lowered        # mood
        assert "coffee table" in lowered  # objects

    def test_description_reads_as_sentences(self):
        fields, _ = normalize_fields(RAW)
        desc, _ = generate_description(fields)
        assert desc[0].isupper()
        assert desc.endswith(".")
        assert "  " not in desc

    def test_description_handles_sparse_fields(self):
        desc, _ = generate_description({"primary_subject": "Skull"})
        assert desc.lower().startswith("a skull")
        assert desc.endswith(".")


class TestCategory:
    def test_furniture_beats_generic_art(self):
        fields, _ = normalize_fields(RAW)
        category, conf = infer_category(fields)
        assert category == "Furniture"
        assert conf > 0

    def test_category_none_when_nothing_known(self):
        category, conf = infer_category({})
        assert category is None
        assert conf == 0.0


class TestTags:
    def test_tag_count_within_range(self):
        fields, _ = normalize_fields(RAW)
        fields["dominant_colors"] = STATS.colors
        tags = generate_tags(fields, STATS)
        assert 10 <= len(tags) <= 30

    def test_tags_are_slugs_and_unique(self):
        fields, _ = normalize_fields(RAW)
        tags = generate_tags(fields, STATS)
        assert len(tags) == len(set(tags))
        for tag in tags:
            assert tag == tag.lower()
            assert " " not in tag

    def test_minimum_tags_even_for_empty_input(self):
        assert len(generate_tags({}, STATS)) >= 10


class TestAttributes:
    def test_generates_well_beyond_six_traits(self):
        fields, _ = normalize_fields(RAW)
        fields["dominant_colors"] = STATS.colors
        fields["category"] = "Furniture"
        fields["visual_complexity"] = STATS.complexity
        attrs = generate_attributes(fields, STATS)
        assert len(attrs) > 6

    def test_trait_types_are_unique(self):
        fields, _ = normalize_fields(RAW)
        fields["dominant_colors"] = STATS.colors
        attrs = generate_attributes(fields, STATS)
        trait_types = [a.trait_type for a in attrs]
        assert len(trait_types) == len(set(trait_types))

    def test_numeric_traits_declare_display_type(self):
        fields, _ = normalize_fields(RAW)
        fields["dominant_colors"] = STATS.colors
        attrs = generate_attributes(fields, STATS)
        numeric = [a for a in attrs if a.display_type == "number"]
        assert {a.trait_type for a in numeric} == {"Object Count", "Color Count"}

    def test_missing_fields_produce_no_empty_traits(self):
        attrs = generate_attributes({}, STATS)
        assert all(a.value not in ("", "Unknown", None) for a in attrs)


class TestMetadataAssembly:
    def test_full_metadata_is_populated(self):
        meta = generate_metadata(RAW, STATS)
        assert meta.title == meta.name
        assert meta.category == "Furniture"
        assert meta.style == "Classic"
        assert meta.primary_subject == "Sofa Set"
        assert meta.dominant_colors == ["Light Blue", "Brown", "White"]
        assert meta.visual_complexity == "Medium"
        assert meta.composition  # derived, never absent
        assert len(meta.tags) >= 10
        assert len(meta.attributes) > 6

    def test_every_emitted_field_has_confidence(self):
        meta = generate_metadata(RAW, STATS)
        tracked = [
            "title", "description", "category", "style", "primary_subject",
            "scene", "environment", "lighting", "mood", "composition",
            "perspective", "texture", "pattern", "visual_complexity",
            "art_medium", "dominant_colors", "materials", "objects", "tags",
            "attributes",
        ]
        for field in tracked:
            assert field in meta.confidence, f"missing confidence for {field}"

    def test_confidence_never_exceeds_ceiling(self):
        meta = generate_metadata(RAW, STATS)
        assert all(0 < v <= 0.9 for v in meta.confidence.values())

    def test_measured_fields_outrank_generated_ones(self):
        meta = generate_metadata(RAW, STATS)
        # Complexity is measured from pixels; style is a model guess.
        assert meta.confidence["visual_complexity"] > meta.confidence["style"]

    def test_empty_model_output_still_yields_valid_metadata(self):
        meta = generate_metadata({}, STATS)
        assert meta.title
        assert meta.description
        assert meta.dominant_colors == ["Light Blue", "Brown", "White"]
        assert len(meta.tags) >= 10

    def test_deterministic(self):
        assert generate_metadata(RAW, STATS) == generate_metadata(RAW, STATS)
