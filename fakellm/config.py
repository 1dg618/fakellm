"""Configuration loading and validation.

Reads fakellm.yaml, validates it with Pydantic, and precomputes the per-rule
fields the matcher needs (lowercased needles, compiled regexes). Doing this
once at load time saves work on every request and surfaces typos in `when:`
keys as a clear error instead of a silent never-match.

The on-disk YAML format is unchanged from earlier versions — existing configs
continue to load without modification.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class MatcherWhen(BaseModel):
    """Conditions on a rule. Unknown keys are rejected to catch YAML typos.

    Header matchers are stored in `headers` (a dict). In YAML they appear as
    `header.x-some-name: value`; we flatten those into the dict in the
    validator below.
    """

    # Reject unknown fields. This is the main reason for using Pydantic here:
    # a typo like `messages_contains` would otherwise silently never match.
    model_config = ConfigDict(extra="forbid")

    messages_contain: str | None = None
    model_matches: str | None = None
    tools_include: str | None = None
    turn: int | None = None
    turn_in: list[int] | None = None
    previous_message_role: str | None = None
    previous_message_contains: str | None = None
    tool_result_contains: str | None = None

    # Populated by the pre-validator from `header.*` keys; not a YAML field
    # users write directly.
    headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("turn_in")
    @classmethod
    def _validate_turn_in(cls, v: list[int] | None) -> list[int] | None:
        if v is None:
            return v
        if len(v) != 2:
            raise ValueError("turn_in must be a [low, high] pair")
        if v[0] > v[1]:
            raise ValueError(f"turn_in low ({v[0]}) is greater than high ({v[1]})")
        return v


class Respond(BaseModel):
    """The response a rule produces."""

    model_config = ConfigDict(extra="forbid")

    status: int = 200
    content: str | None = None
    error: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    # Back-compat shim: older code (and existing tests) access fields with
    # `respond["content"]`. Proxying to attribute access lets them keep
    # working without forcing a rewrite of the test suite. Attribute access
    # remains the canonical API for new code.
    def __getitem__(self, key: str) -> Any:
        if key not in type(self).model_fields:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in type(self).model_fields:
            return default
        value = getattr(self, key)
        return value if value is not None else default


class Rule(BaseModel):
    """One rule: a matcher and the response to produce when it matches."""

    # Allow arbitrary types so we can attach the compiled regex below.
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    when: MatcherWhen = Field(default_factory=MatcherWhen)
    respond: Respond = Field(default_factory=Respond)

    # ---- Precomputed fields (populated by load_config, not from YAML) ----
    # These exist so the matcher can do O(1) lookups instead of re-lowering
    # strings and re-compiling regexes on every request.
    _messages_contain_lower: str | None = None
    _model_regex: re.Pattern[str] | None = None
    _previous_message_contains_lower: str | None = None
    _tool_result_contains_lower: str | None = None

    # Back-compat shim: see Respond.__getitem__ above. Same reasoning.
    def __getitem__(self, key: str) -> Any:
        if key not in type(self).model_fields:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in type(self).model_fields:
            return default
        value = getattr(self, key)
        return value if value is not None else default


class Config(BaseModel):
    """Top-level config: a list of rules and a defaults dict."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rules: list[Rule] = Field(default_factory=list)
    defaults: dict[str, Any] = Field(default_factory=dict)

    @field_validator("rules", mode="before")
    @classmethod
    def _accept_raw_dicts(cls, v: Any) -> Any:
        """Allow Config(rules=[{...}, {...}]) with raw dicts.

        load_config() always passes already-built Rule objects, but tests
        (and some older callers) construct Config directly with dict-shaped
        rules. We route those through the same _build_rule path so they
        get the precomputed fields too — without this, dict-built rules
        would match correctly but slowly, since the matcher's hot path
        reads the precomputed fields.
        """
        if not isinstance(v, list):
            return v
        out = []
        for i, item in enumerate(v):
            if isinstance(item, Rule):
                out.append(item)
            elif isinstance(item, dict):
                out.append(_build_rule(item, index=i))
            else:
                out.append(item)  # let Pydantic raise its own error
        return out


def load_config(path: str | Path = "fakellm.yaml") -> Config:
    """Load and validate config from a YAML file.

    Returns an empty Config if the file is missing. Raises ValueError with a
    readable message if the YAML is malformed or has unknown keys.
    """
    p = Path(path)
    if not p.exists():
        return Config()

    with p.open() as f:
        raw = yaml.safe_load(f) or {}

    raw_rules = raw.get("rules", [])
    if not isinstance(raw_rules, list):
        raise ValueError(f"'rules' must be a list, got {type(raw_rules).__name__}")

    processed: list[Rule] = []
    for i, r in enumerate(raw_rules):
        if not isinstance(r, dict):
            raise ValueError(f"rule[{i}] must be a mapping, got {type(r).__name__}")
        try:
            processed.append(_build_rule(r, index=i))
        except ValidationError as e:
            # Pydantic's default message is long; trim to the first error for
            # config errors — users want to know which field, not a stack.
            err = e.errors()[0]
            field = ".".join(str(p) for p in err["loc"]) or "(root)"
            raise ValueError(
                f"rule[{i}] ({r.get('name', '<unnamed>')}): "
                f"field '{field}': {err['msg']}"
            ) from e

    return Config(rules=processed, defaults=raw.get("defaults", {}) or {})


def _build_rule(raw: dict[str, Any], index: int) -> Rule:
    """Validate one raw rule dict and precompute matcher fields."""
    raw_when = dict(raw.get("when") or {})

    # Extract header.* keys into a single `headers` dict before validation.
    # We do this here (rather than in a Pydantic validator) so the
    # `extra="forbid"` check on MatcherWhen still works for genuine typos.
    headers: dict[str, str] = {}
    for k in list(raw_when.keys()):
        if k.startswith("header."):
            headers[k[len("header."):].lower()] = str(raw_when.pop(k))
    if headers:
        raw_when["headers"] = headers

    rule = Rule(
        name=raw.get("name", f"rule-{index}"),
        when=MatcherWhen(**raw_when),
        respond=Respond(**(raw.get("respond") or {})),
    )

    # Precompute the lowercased / compiled fields used on the matcher hot path.
    when = rule.when
    if when.messages_contain is not None:
        rule._messages_contain_lower = when.messages_contain.lower()
    if when.model_matches is not None:
        # Preserve the original glob behavior: `*` is wildcard, anchored.
        # re.escape everything else so `gpt-4*` doesn't break on the dash.
        parts = [re.escape(p) for p in when.model_matches.split("*")]
        pattern = ".*".join(parts)
        rule._model_regex = re.compile(f"^{pattern}$")
    if when.previous_message_contains is not None:
        rule._previous_message_contains_lower = when.previous_message_contains.lower()
    if when.tool_result_contains is not None:
        rule._tool_result_contains_lower = when.tool_result_contains.lower()

    return rule


def normalize_rule(rule: Any) -> Rule | None:
    """Coerce a rule-shaped value into a Rule.

    Accepts None, a Rule, or a dict in the YAML shape (`{"name": ..., "when":
    {...}, "respond": {...}}` or just `{"respond": {...}}`). Returns a fully
    precomputed Rule, or None if the input was None.

    This exists so build_response / build_stream can take either a Rule
    (the canonical type, produced by load_config) or a raw dict (used by
    older callers and the test suite that hand-builds rules). The
    conversion is cheap and runs the same precomputation as load_config,
    so downstream code can always assume it's got a Rule.
    """
    if rule is None or isinstance(rule, Rule):
        return rule
    if isinstance(rule, dict):
        # Tests sometimes pass `{"respond": {...}}` with no name. Fill in a
        # placeholder rather than rejecting — name is only used for stats,
        # not matching behavior.
        if "name" not in rule:
            rule = {"name": "<unnamed>", **rule}
        return _build_rule(rule, index=0)
    raise TypeError(f"Expected Rule, dict, or None; got {type(rule).__name__}")
