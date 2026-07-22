from pydantic import BaseModel, Field


class Attribute(BaseModel):
    """OpenSea-compatible trait. `display_type` is only set for numeric traits."""

    trait_type: str
    value: str | int | float
    display_type: str | None = None


class NFTMetadata(BaseModel):
    # ERC-721 / OpenSea core. `name` mirrors `title` so the payload can be
    # pinned as token metadata without a translation step.
    name: str
    title: str
    description: str

    # Semantic facets — the fields a creator would otherwise fill by hand.
    category: str | None = None
    style: str | None = None
    primary_subject: str | None = None
    secondary_subjects: list[str] = Field(default_factory=list)
    scene: str | None = None
    environment: str | None = None
    objects: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    dominant_colors: list[str] = Field(default_factory=list)
    lighting: str | None = None
    mood: str | None = None
    composition: str | None = None
    perspective: str | None = None
    texture: str | None = None
    pattern: str | None = None
    visual_complexity: str | None = None
    art_medium: str | None = None

    tags: list[str] = Field(default_factory=list)
    attributes: list[Attribute] = Field(default_factory=list)

    # Per-field score in [0, 1]. Anything the creator should double-check
    # scores low; only pixel-measured facts approach the 0.9 ceiling.
    confidence: dict[str, float] = Field(default_factory=dict)


class NFTResponse(BaseModel):
    image_hash: str
    metadata: NFTMetadata
    duration_ms: int | None = None
    cached: bool = False


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
