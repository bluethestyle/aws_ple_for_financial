"""
Schema Registry — single source of truth for dataset schemas.

Loads schema definitions from YAML, validates DataFrames, tracks schema
evolution, and identifies PII columns.

Example YAML schema::

    sources:
      user_events:
        columns:
          user_id:
            type: string
            pii: true
            nullable: false
          event_ts:
            type: timestamp
            nullable: false
          amount:
            type: float64
            nullable: true
            min: 0.0
          category:
            type: string
            nullable: true
            allowed_values: ["A", "B", "C"]
        primary_key: [user_id, event_ts]
        version: 2
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ColumnSpec:
    """Specification for a single column."""

    name: str
    dtype: str  # string, int64, float64, float32, bool, timestamp, date
    nullable: bool = True
    pii: bool = False
    min: Optional[float] = None
    max: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    description: str = ""


@dataclass
class SourceSchema:
    """Schema definition for a single data source / table."""

    name: str
    columns: Dict[str, ColumnSpec] = field(default_factory=dict)
    primary_key: List[str] = field(default_factory=list)
    version: int = 1
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # ── helpers ────────────────────────────────────────────────────────

    @property
    def column_names(self) -> List[str]:
        return list(self.columns.keys())

    @property
    def pii_columns(self) -> List[str]:
        return [c.name for c in self.columns.values() if c.pii]

    @property
    def numeric_columns(self) -> List[str]:
        _NUM_TYPES = {"int64", "int32", "float64", "float32", "double", "int"}
        return [c.name for c in self.columns.values() if c.dtype in _NUM_TYPES]

    @property
    def categorical_columns(self) -> List[str]:
        return [
            c.name
            for c in self.columns.values()
            if c.dtype == "string" and c.allowed_values is not None
        ]

    @property
    def timestamp_columns(self) -> List[str]:
        return [
            c.name
            for c in self.columns.values()
            if c.dtype in {"timestamp", "date"}
        ]


# ──────────────────────────────────────────────────────────────────────
# Schema Registry
# ──────────────────────────────────────────────────────────────────────


class SchemaRegistry:
    """
    Central registry that stores and manages dataset schemas.

    Schemas can be loaded from YAML files or registered programmatically.
    The registry supports schema evolution with backward-compatibility
    checks.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to a YAML file (or directory of YAML files) containing schema
        definitions.  If ``None``, the registry starts empty and schemas
        must be registered via :meth:`register`.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._schemas: Dict[str, SourceSchema] = {}
        self._history: Dict[str, List[SourceSchema]] = {}
        if config_path is not None:
            self.load(config_path)

    # ── I/O ────────────────────────────────────────────────────────────

    def load(self, path: str) -> "SchemaRegistry":
        """Load schema definitions from a YAML file or a directory of YAMLs.

        Parameters
        ----------
        path : str
            File path or directory path.  When a directory is given every
            ``*.yaml`` and ``*.yml`` file inside it is loaded.

        Returns
        -------
        SchemaRegistry
            ``self``, for chaining.
        """
        p = Path(path)
        if p.is_dir():
            for f in sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml")):
                self._load_file(f)
        elif p.is_file():
            self._load_file(p)
        else:
            raise FileNotFoundError(f"Schema path not found: {path}")
        return self

    def _load_file(self, path: Path) -> None:
        """Parse a single YAML schema file."""
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        if raw is None:
            logger.warning("Empty schema file: %s", path)
            return

        sources = raw.get("sources", raw)  # top-level key is optional
        for source_name, source_def in sources.items():
            schema = self._parse_source(source_name, source_def)
            self.register(schema)
            logger.info(
                "Loaded schema '%s' v%d (%d columns) from %s",
                source_name, schema.version, len(schema.columns), path,
            )

    @staticmethod
    def _parse_source(name: str, raw: Dict[str, Any]) -> SourceSchema:
        """Convert raw dict to a ``SourceSchema`` object."""
        columns: Dict[str, ColumnSpec] = {}
        for col_name, col_def in raw.get("columns", {}).items():
            if isinstance(col_def, str):
                # shorthand: ``column_name: float64``
                col_def = {"type": col_def}
            columns[col_name] = ColumnSpec(
                name=col_name,
                dtype=col_def.get("type", "string"),
                nullable=col_def.get("nullable", True),
                pii=col_def.get("pii", False),
                min=col_def.get("min"),
                max=col_def.get("max"),
                allowed_values=col_def.get("allowed_values"),
                description=col_def.get("description", ""),
            )
        return SourceSchema(
            name=name,
            columns=columns,
            primary_key=raw.get("primary_key", []),
            version=raw.get("version", 1),
            description=raw.get("description", ""),
            tags=raw.get("tags", []),
        )

    def save(self, path: str) -> None:
        """Serialise all schemas to a single YAML file.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        out: Dict[str, Any] = {"sources": {}}
        for name, schema in self._schemas.items():
            cols: Dict[str, Any] = {}
            for col in schema.columns.values():
                entry: Dict[str, Any] = {"type": col.dtype}
                if not col.nullable:
                    entry["nullable"] = False
                if col.pii:
                    entry["pii"] = True
                if col.min is not None:
                    entry["min"] = col.min
                if col.max is not None:
                    entry["max"] = col.max
                if col.allowed_values is not None:
                    entry["allowed_values"] = col.allowed_values
                if col.description:
                    entry["description"] = col.description
                cols[col.name] = entry
            source_dict: Dict[str, Any] = {
                "columns": cols,
                "version": schema.version,
            }
            if schema.primary_key:
                source_dict["primary_key"] = schema.primary_key
            if schema.description:
                source_dict["description"] = schema.description
            if schema.tags:
                source_dict["tags"] = schema.tags
            out["sources"][name] = source_dict

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(out, fh, default_flow_style=False, sort_keys=False)
        logger.info("Saved %d schemas to %s", len(self._schemas), path)

    # ── Registration / lookup ──────────────────────────────────────────

    def register(self, schema: SourceSchema) -> None:
        """Register (or update) a source schema.

        If a schema with the same name already exists, backward
        compatibility is checked before overwriting.

        Parameters
        ----------
        schema : SourceSchema
            The schema to register.

        Raises
        ------
        SchemaEvolutionError
            When the new schema breaks backward compatibility (removed
            non-nullable columns).
        """
        if schema.name in self._schemas:
            self._check_compatibility(self._schemas[schema.name], schema)
            self._history.setdefault(schema.name, []).append(
                copy.deepcopy(self._schemas[schema.name])
            )
        self._schemas[schema.name] = schema

    def get(self, name: str) -> SourceSchema:
        """Return the schema for *name*.

        Raises ``KeyError`` if not found.
        """
        if name not in self._schemas:
            raise KeyError(
                f"Schema '{name}' not found. "
                f"Available: {list(self._schemas.keys())}"
            )
        return self._schemas[name]

    def list_sources(self) -> List[str]:
        """Return a sorted list of registered source names."""
        return sorted(self._schemas.keys())

    def has(self, name: str) -> bool:
        """Return ``True`` if a schema with *name* is registered."""
        return name in self._schemas

    # ── PII helpers ────────────────────────────────────────────────────

    def pii_columns(self, source: str) -> List[str]:
        """Return the list of PII-marked columns for *source*."""
        return self.get(source).pii_columns

    def all_pii_columns(self) -> Dict[str, List[str]]:
        """Return a mapping ``{source: [pii_col, ...]}`` across all sources."""
        return {
            name: schema.pii_columns
            for name, schema in self._schemas.items()
            if schema.pii_columns
        }

    # ── Validation ─────────────────────────────────────────────────────

    def validate_dataframe(
        self, source: str, df: "pandas.DataFrame"  # noqa: F821
    ) -> Tuple[bool, List[str]]:
        """Validate a pandas DataFrame against the registered schema.

        Parameters
        ----------
        source : str
            Name of the registered source schema.
        df : pandas.DataFrame
            DataFrame to validate.

        Returns
        -------
        (valid, errors) : tuple[bool, list[str]]
            ``valid`` is ``True`` when no errors are found.
        """
        schema = self.get(source)
        errors: List[str] = []

        # check required columns
        df_cols: Set[str] = set(df.columns)
        for col_name, col_spec in schema.columns.items():
            if col_name not in df_cols:
                if not col_spec.nullable:
                    errors.append(
                        f"Required column '{col_name}' missing from DataFrame"
                    )
                continue

            # nullability
            if not col_spec.nullable and df[col_name].isna().any():
                errors.append(
                    f"Column '{col_name}' contains NULLs but is not nullable"
                )

        return (len(errors) == 0, errors)

    # ── Schema evolution ───────────────────────────────────────────────

    @staticmethod
    def _check_compatibility(
        old: SourceSchema, new: SourceSchema
    ) -> None:
        """Ensure backward compatibility between schema versions.

        Rules:
        * Adding new columns is always allowed.
        * Removing a **non-nullable** column that existed in the old
          schema is forbidden (breaking change).
        * Changing a column from non-nullable to nullable is allowed.

        Raises
        ------
        SchemaEvolutionError
            On a breaking change.
        """
        removed = set(old.columns.keys()) - set(new.columns.keys())
        breaking: List[str] = []
        for col_name in removed:
            if not old.columns[col_name].nullable:
                breaking.append(col_name)

        if breaking:
            raise SchemaEvolutionError(
                f"Breaking schema change for '{old.name}': "
                f"non-nullable columns removed: {breaking}"
            )

    def history(self, name: str) -> List[SourceSchema]:
        """Return previous versions of the schema (oldest first)."""
        return list(self._history.get(name, []))

    def evolve(
        self,
        source: str,
        *,
        add_columns: Optional[Dict[str, Dict[str, Any]]] = None,
        remove_columns: Optional[List[str]] = None,
    ) -> SourceSchema:
        """Create a new version of a schema with column additions/removals.

        Parameters
        ----------
        source : str
            Name of the existing schema.
        add_columns : dict, optional
            Mapping ``{col_name: {type: ..., ...}}`` for new columns.
        remove_columns : list[str], optional
            Column names to drop.  Removing non-nullable columns raises.

        Returns
        -------
        SourceSchema
            The new (registered) schema.
        """
        old = self.get(source)
        new_cols = dict(old.columns)

        if add_columns:
            for col_name, col_def in add_columns.items():
                new_cols[col_name] = ColumnSpec(
                    name=col_name,
                    dtype=col_def.get("type", "string"),
                    nullable=col_def.get("nullable", True),
                    pii=col_def.get("pii", False),
                    min=col_def.get("min"),
                    max=col_def.get("max"),
                    allowed_values=col_def.get("allowed_values"),
                    description=col_def.get("description", ""),
                )

        if remove_columns:
            for col_name in remove_columns:
                if col_name in new_cols:
                    del new_cols[col_name]

        new_schema = SourceSchema(
            name=old.name,
            columns=new_cols,
            primary_key=old.primary_key,
            version=old.version + 1,
            description=old.description,
            tags=list(old.tags),
        )
        self.register(new_schema)  # compatibility is checked inside
        return new_schema

    # ── Dunder ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._schemas)

    def __contains__(self, name: str) -> bool:
        return name in self._schemas

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SchemaRegistry(sources={list(self._schemas.keys())})"
        )


# ──────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────


class SchemaEvolutionError(Exception):
    """Raised when a schema evolution breaks backward compatibility."""
