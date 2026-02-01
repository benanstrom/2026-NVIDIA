from __future__ import annotations


def get_seeder(name: str):
    """Seeder registry."""

    key = name.lower().strip()
    if key == "random":
        from .random_seeder import generate_seeds

        return generate_seeds
    if key == "qaoa":
        from .qaoa_seeder import generate_seeds

        return generate_seeds
    if key == "dcqo":
        from .dcqo_seeder import generate_seeds

        return generate_seeds
    if key in ("dcqo_plus", "dcqo+", "dcqoplus"):
        from .dcqo_plus_seeder import generate_seeds

        return generate_seeds
    if key == "pce":
        from .pce_seeder import generate_seeds

        return generate_seeds
    raise KeyError(f"Unknown seeder: {name}")


def list_seeders() -> list[str]:
    return ["random", "qaoa", "dcqo", "dcqo_plus", "pce"]
