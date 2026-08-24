"""Validated, serializable configuration contracts for simcat workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a finite number")
    return number


def _unit_interval(low: float, high: float, name: str) -> None:
    if not 0 <= low <= high <= 1:
        raise ValueError(f"{name} must satisfy 0 <= minimum <= maximum <= 1")


@dataclass(frozen=True)
class TreeConfig:
    """Species-tree identity and the feature row order it defines."""

    newick: str
    tip_order: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.newick, str) or not self.newick.strip():
            raise ValueError("newick must be a non-empty string")
        tips = tuple(str(tip) for tip in self.tip_order)
        if len(tips) < 4:
            raise ValueError("tip_order must contain at least four tips")
        if len(set(tips)) != len(tips):
            raise ValueError("tip_order must contain unique tip names")
        try:
            import toytree

            parsed_tips = tuple(
                str(tip) for tip in toytree.tree(self.newick).get_tip_labels()
            )
        except Exception as exc:
            raise ValueError("newick must describe a valid species tree") from exc
        if parsed_tips != tips:
            raise ValueError(
                "tip_order must exactly match the leaf order encoded by newick"
            )
        object.__setattr__(self, "tip_order", tips)

    @classmethod
    def from_tree(cls, tree: Any) -> "TreeConfig":
        """Create a config from a toytree-like object without importing it."""
        return cls(
            newick=str(tree.write()),
            tip_order=tuple(str(tip) for tip in tree.get_tip_labels()),
        )


@dataclass(frozen=True)
class ParameterRanges:
    """Validated simulation parameter ranges."""

    Ne_min: float = 10_000
    Ne_max: float = 100_000
    admix_prop_min: float = 0.05
    admix_prop_max: float = 0.50
    admix_edge_min: float = 0.50
    admix_edge_max: float = 0.50
    node_slide_prop: float = 0.25

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _finite_number(getattr(self, name), name))
        if self.Ne_min <= 0 or self.Ne_max < self.Ne_min:
            raise ValueError("Ne values must satisfy 0 < Ne_min <= Ne_max")
        _unit_interval(
            self.admix_prop_min, self.admix_prop_max, "admixture proportions"
        )
        _unit_interval(
            self.admix_edge_min, self.admix_edge_max, "admixture edge positions"
        )
        if not 0 <= self.node_slide_prop <= 1:
            raise ValueError("node_slide_prop must be between 0 and 1")


@dataclass(frozen=True)
class SubstitutionModelConfig:
    """JC69 or GTR substitution-model parameters."""

    model: str = "JC69"
    rate_vector: Optional[Tuple[float, ...]] = None
    pi_vector: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        model = self.model.upper()
        object.__setattr__(self, "model", model)
        if model not in {"JC69", "GTR"}:
            raise ValueError("model must be 'JC69' or 'GTR'")
        if model == "JC69":
            if self.rate_vector is not None or self.pi_vector is not None:
                raise ValueError("JC69 does not accept rate_vector or pi_vector")
            return
        if self.rate_vector is None or self.pi_vector is None:
            raise ValueError("GTR requires both rate_vector and pi_vector")
        rates = tuple(_finite_number(item, "GTR rate") for item in self.rate_vector)
        frequencies = tuple(
            _finite_number(item, "GTR frequency") for item in self.pi_vector
        )
        if len(rates) != 6 or any(item <= 0 for item in rates):
            raise ValueError("rate_vector must contain six positive finite GTR rates")
        if len(frequencies) != 4 or any(item <= 0 for item in frequencies):
            raise ValueError("pi_vector must contain four positive finite frequencies")
        if abs(sum(frequencies) - 1.0) > 1e-8:
            raise ValueError("pi_vector base frequencies must sum to one")
        object.__setattr__(self, "rate_vector", rates)
        object.__setattr__(self, "pi_vector", frequencies)


@dataclass(frozen=True)
class RNGConfig:
    """Master random seed contract."""

    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.seed is not None:
            if isinstance(self.seed, bool) or int(self.seed) != self.seed:
                raise ValueError("seed must be a non-negative integer or None")
            if self.seed < 0 or self.seed > 2**32 - 1:
                raise ValueError("seed must be between 0 and 2**32 - 1")
            object.__setattr__(self, "seed", int(self.seed))


@dataclass(frozen=True)
class StorageConfig:
    """Names and paths for a database artifact set."""

    name: str
    workdir: Path = Path("databases")
    force: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if Path(self.name).name != self.name or self.name in {".", ".."}:
            raise ValueError("name must not contain path components")
        object.__setattr__(self, "workdir", Path(self.workdir))
        if not isinstance(self.force, bool):
            raise ValueError("force must be a boolean")


@dataclass(frozen=True)
class DatabaseConfig:
    """Complete configuration needed to generate Phase 2 labels artifacts."""

    tree: TreeConfig
    storage: StorageConfig
    parameters: ParameterRanges = field(default_factory=ParameterRanges)
    substitution: SubstitutionModelConfig = field(
        default_factory=SubstitutionModelConfig
    )
    rng: RNGConfig = field(default_factory=RNGConfig)
    nrows: int = 100
    nsnps: int = 20_000
    existing_admix_edges: Tuple[Tuple[int, int], ...] = ()
    exclude_sisters: bool = False
    quiet: bool = False

    def __post_init__(self) -> None:
        for name in ("nrows", "nsnps"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        edges = tuple(tuple(int(node) for node in edge) for edge in self.existing_admix_edges)
        if any(len(edge) != 2 or min(edge) < 0 for edge in edges):
            raise ValueError("existing_admix_edges must contain non-negative pairs")
        object.__setattr__(self, "existing_admix_edges", edges)
        for name in ("exclude_sisters", "quiet"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["storage"]["workdir"] = str(self.storage.workdir)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatabaseConfig":
        return cls(
            tree=TreeConfig(**data["tree"]),
            storage=StorageConfig(**data["storage"]),
            parameters=ParameterRanges(**data.get("parameters", {})),
            substitution=SubstitutionModelConfig(**data.get("substitution", {})),
            rng=RNGConfig(**data.get("rng", {})),
            nrows=data.get("nrows", 100),
            nsnps=data.get("nsnps", 20_000),
            existing_admix_edges=tuple(data.get("existing_admix_edges", ())),
            exclude_sisters=data.get("exclude_sisters", False),
            quiet=data.get("quiet", False),
        )


@dataclass(frozen=True)
class TrainingConfig:
    """Validated data-selection and training configuration."""

    input_name: str
    output_name: str
    directory: Path
    prop_training: float = 0.9
    exclude_sisters: bool = True
    exclude_magnitude: float = 0.1
    batch_size: int = 20
    num_epochs: int = 1
    seed: Optional[int] = None
    feature_normalization: str = "per_quartet_max"

    def __post_init__(self) -> None:
        for name in ("input_name", "output_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or Path(value).name != value:
                raise ValueError(f"{name} must be a non-empty artifact name")
        object.__setattr__(self, "directory", Path(self.directory))
        if not 0 < self.prop_training < 1:
            raise ValueError("prop_training must be strictly between 0 and 1")
        object.__setattr__(
            self,
            "exclude_magnitude",
            _finite_number(self.exclude_magnitude, "exclude_magnitude"),
        )
        if self.exclude_magnitude < 0:
            raise ValueError("exclude_magnitude must be non-negative")
        if not isinstance(self.exclude_sisters, bool):
            raise ValueError("exclude_sisters must be a boolean")
        for name in ("batch_size", "num_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        RNGConfig(self.seed)
        if self.feature_normalization != "per_quartet_max":
            raise ValueError(
                "feature_normalization must be 'per_quartet_max' in schema 1"
            )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["directory"] = str(self.directory)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingConfig":
        """Restore a training contract from its JSON-compatible mapping."""
        return cls(**dict(data))
