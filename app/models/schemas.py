from pydantic import BaseModel


class Attribute(BaseModel):
    trait_type: str
    value: str


class NFTMetadata(BaseModel):
    name: str
    description: str
    attributes: list[Attribute]


class NFTResponse(BaseModel):
    image_hash: str
    metadata: NFTMetadata


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
