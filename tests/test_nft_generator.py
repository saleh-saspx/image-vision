from app.services.imagestats import ImageStats
from app.services.nft_generator import (
    MAX_TAGS,
    MIN_CONFIDENCE,
    MIN_TAGS,
    _apply_confidence_floor,
    generate_attributes,
    generate_description,
    generate_metadata,
    generate_tags,
    generate_title,
    infer_category,
    infer_environment,
    infer_materials,
    infer_scene,
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


class TestPlaceholderRejection:
    """The model often echoes the schema back. None of it may reach the user."""

    def test_echoed_field_names_are_dropped(self):
        fields, conf = normalize_fields({
            "primary_subject": "Main Subject",
            "scene": "scene",
            "style": "style",
            "mood": "mood",
            "art_medium": "medium",
        })
        assert "primary_subject" not in fields
        assert "scene" not in fields
        assert "style" not in fields
        assert "mood" not in fields
        assert conf == {}

    def test_generic_nouns_are_dropped(self):
        for junk in ("object", "Generic Object", "artwork", "image", "thing", "item"):
            fields, _ = normalize_fields({"primary_subject": junk})
            assert "primary_subject" not in fields, f"{junk!r} leaked through"

    def test_numbered_placeholders_are_dropped(self):
        for junk in ("Object 1", "item #2", "Subject A", "figure 12"):
            fields, _ = normalize_fields({"primary_subject": junk})
            assert "primary_subject" not in fields, f"{junk!r} leaked through"

    def test_qualified_placeholders_are_dropped(self):
        # Enumerating exact phrases does not scale; the shape is what matters.
        for junk in (
            "Other Notable Subjects", "Visible Objects", "Generic Object",
            "key element", "various things", "unidentified figure",
        ):
            fields, _ = normalize_fields({"primary_subject": junk})
            assert "primary_subject" not in fields, f"{junk!r} leaked through"

    def test_uncopied_prompt_slots_are_dropped(self):
        fields, _ = normalize_fields({
            "primary_subject": "<the one main object>",
            "scene": "<room or place>",
            "objects": ["<every object you can see>", "Sofa"],
        })
        assert "primary_subject" not in fields
        assert "scene" not in fields
        assert fields["objects"] == ["Sofa"]

    def test_real_multiword_subjects_survive_the_shape_filter(self):
        for real in (
            "Chess Piece", "Portrait of a Woman", "Mountain Cabin",
            "Sports Car", "Fantasy Castle", "Robot Helmet",
        ):
            fields, _ = normalize_fields({"primary_subject": real})
            assert fields.get("primary_subject"), f"{real!r} was wrongly filtered"

    def test_placeholders_are_stripped_from_lists(self):
        fields, _ = normalize_fields({
            "objects": ["Sofa", "Object", "Coffee Table", "thing", "Item 2"],
        })
        assert fields["objects"] == ["Sofa", "Coffee Table"]

    def test_real_values_are_not_over_filtered(self):
        fields, _ = normalize_fields({
            "primary_subject": "Robot Helmet",
            "scene": "living room",
            "style": "cyberpunk",
        })
        assert fields["primary_subject"] == "Robot Helmet"
        assert fields["scene"] == "Living Room"
        assert fields["style"] == "Cyberpunk"

    def test_placeholder_subject_never_reaches_the_title(self):
        meta = generate_metadata({"primary_subject": "Main Subject"}, STATS)
        assert "main subject" not in meta.title.lower()
        assert "subject" not in meta.title.lower()

    def test_no_placeholder_survives_into_tags_or_traits(self):
        meta = generate_metadata(
            {"primary_subject": "Object", "objects": ["Thing", "Item 1"], "scene": "scene"},
            STATS,
        )
        banned = {"object", "thing", "item", "subject", "scene", "artwork", "image"}
        assert not (banned & set(meta.tags))
        assert not any(str(a.value).lower() in banned for a in meta.attributes)


class TestGapFilling:
    """Fields that are visually obvious must not come back null."""

    def test_scene_deduced_from_objects(self):
        fields, _ = normalize_fields({"objects": ["Sofa", "Coffee Table", "Cushion"]})
        scene, score = infer_scene(fields)
        assert scene == "Living Room"
        assert score > 0

    def test_scene_deduction_needs_a_decisive_object(self):
        fields, _ = normalize_fields({"objects": ["Cup"]})
        assert infer_scene(fields)[0] is None

    def test_environment_deduced_from_scene(self):
        assert infer_environment({"scene": "Living Room"})[0] == "Indoor"
        assert infer_environment({"scene": "Beach"})[0] == "Outdoor"
        assert infer_environment({"scene": "Space"})[0] == "Outer Space"
        assert infer_environment({})[0] is None

    def test_materials_deduced_from_wording(self):
        fields, _ = normalize_fields({"primary_subject": "vintage wooden armchair"})
        materials, score = infer_materials(fields)
        assert materials == ["Wood"]
        assert score > 0

    def test_colour_words_are_not_read_as_materials(self):
        # "Golden Retriever" is a dog, not a metal object.
        fields, _ = normalize_fields({"primary_subject": "golden retriever"})
        assert infer_materials(fields)[0] == []

    def test_deduction_never_overwrites_the_model(self):
        raw = dict(RAW)
        raw["scene"] = "kitchen"  # model says kitchen despite sofa objects
        meta = generate_metadata(raw, STATS)
        assert meta.scene == "Kitchen"

    def test_obvious_fields_are_populated_from_a_sparse_response(self):
        """The model named only the subject and objects — everything else here
        is deducible, and returning null for it was the reported bug."""
        meta = generate_metadata(
            {"primary_subject": "a wooden sofa set", "objects": ["Sofa", "Coffee Table"]},
            STATS,
        )
        assert meta.scene == "Living Room"
        assert meta.environment == "Indoor"
        assert meta.materials == ["Wood"]
        assert meta.category == "Furniture"
        assert meta.composition is not None
        assert meta.visual_complexity is not None

    def test_lighting_falls_back_to_pixel_measurement(self):
        dark = ImageStats(
            colors=["Black"], color_weights=[0.9], complexity="Simple",
            orientation="Square", lighting="Low Light", lighting_confidence=0.58,
        )
        meta = generate_metadata({"primary_subject": "Skull"}, dark)
        assert meta.lighting == "Low Light"
        assert meta.confidence["lighting"] == 0.58

    def test_model_lighting_wins_over_the_pixel_guess(self):
        stats = ImageStats(
            colors=["Black"], color_weights=[0.9], complexity="Simple",
            orientation="Square", lighting="Low Light", lighting_confidence=0.58,
        )
        meta = generate_metadata({"lighting": "golden hour"}, stats)
        assert meta.lighting == "Golden Hour"


class TestConfidenceFloor:
    def test_low_confidence_fields_return_null(self):
        fields = {"style": "Cyberpunk", "mood": "Dark"}
        confidence = {"style": 0.78, "mood": 0.2}
        _apply_confidence_floor(fields, confidence)
        assert fields == {"style": "Cyberpunk"}
        assert "mood" not in confidence

    def test_boundary_value_is_kept(self):
        fields = {"style": "Classic"}
        confidence = {"style": MIN_CONFIDENCE}
        _apply_confidence_floor(fields, confidence)
        assert fields == {"style": "Classic"}

    def test_no_emitted_field_scores_below_the_floor(self):
        meta = generate_metadata(RAW, STATS)
        assert all(v >= MIN_CONFIDENCE for v in meta.confidence.values())


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

    def test_material_is_not_repeated_from_the_subject(self):
        fields, _ = normalize_fields(
            {"primary_subject": "vintage wooden armchair", "materials": ["wood"], "style": "rustic"}
        )
        desc, _ = generate_description(fields)
        assert "wood vintage wooden" not in desc.lower()

    def test_color_is_not_repeated_from_the_material(self):
        fields = {
            "primary_subject": "Samurai Helmet",
            "materials": ["Gold"],
            "dominant_colors": ["Gold", "Black"],
        }
        desc, _ = generate_description(fields)
        assert desc.lower().count("gold") == 1

    def test_phrasing_varies_across_a_collection(self):
        subjects = ["Sofa Set", "Crystal Dragon", "Wooden Chair", "Neon Skull", "Ocean Sunset"]
        connectives = set()
        for subject in subjects:
            desc, _ = generate_description(
                {"primary_subject": subject, "scene": "Studio", "dominant_colors": ["Blue"]}
            )
            # The words between the subject and the colour name.
            connectives.add(desc.lower().split(subject.lower())[1].split("blue")[0])
        # A whole collection rendered through one identical template reads as
        # machine output; more than one shape must appear.
        assert len(connectives) > 1

    def test_phrasing_is_stable_for_the_same_image(self):
        fields = {"primary_subject": "Sofa Set", "scene": "Living Room", "dominant_colors": ["Blue"]}
        assert generate_description(fields) == generate_description(fields)

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

    def test_subject_outranks_scene(self):
        # A dragon on a mountain is Fantasy, not Landscape.
        fields, _ = normalize_fields(
            {"primary_subject": "a crystal dragon", "scene": "mountain", "style": "fantasy"}
        )
        assert infer_category(fields)[0] == "Fantasy"

    def test_scene_used_when_no_subject(self):
        fields, _ = normalize_fields({"scene": "living room", "style": "minimalist"})
        assert infer_category(fields)[0] == "Interior Design"

    def test_medium_is_the_last_resort(self):
        fields, _ = normalize_fields({"art_medium": "watercolor"})
        assert infer_category(fields)[0] == "Painting"

    def test_object_is_not_mistaken_for_a_portrait(self):
        fields, _ = normalize_fields({"primary_subject": "a samurai helmet"})
        assert infer_category(fields)[0] == "Fashion"

    def test_every_category_is_in_the_allowed_set(self):
        allowed = {
            "Furniture", "Photography", "Digital Art", "Painting", "Illustration",
            "Architecture", "Interior Design", "Landscape", "Portrait", "Fashion",
            "Gaming", "Fantasy", "Collectible", "Abstract", "Vehicle", "Nature",
            "Animal", "Technology",
        }
        subjects = [
            "sofa", "dragon", "cat", "car", "robot", "mountain", "woman",
            "dress", "building", "flower", "avatar", "geometric shapes",
        ]
        for subject in subjects:
            fields, _ = normalize_fields({"primary_subject": subject})
            category = infer_category(fields)[0]
            assert category in allowed, f"{subject} -> {category}"


class TestTags:
    def test_tag_count_within_range(self):
        fields, _ = normalize_fields(RAW)
        fields["dominant_colors"] = STATS.colors
        tags = generate_tags(fields, STATS)
        assert MIN_TAGS <= len(tags) <= MAX_TAGS

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
        assert {a.trait_type for a in numeric} == {
            "Object Count", "Color Count", "Subject Count",
        }

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
        assert len(meta.tags) >= MIN_TAGS
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
        assert len(meta.tags) >= MIN_TAGS

    def test_deterministic(self):
        assert generate_metadata(RAW, STATS) == generate_metadata(RAW, STATS)
