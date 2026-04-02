from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import RoleName


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key_env: str | None = None
    api_key: str | None = None
    base_url: str
    model_name: str

    @model_validator(mode="after")
    def resolve_api_key(self) -> "LLMConfig":
        if self.api_key is None and self.api_key_env:
            self.api_key = os.getenv(self.api_key_env)
        return self


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_llm: str
    artifacts_dir: str = "artifacts"
    evolution_state_path: str = "artifacts/evolution_state.json"


class RoleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: str
    weight: float = Field(gt=0.0)


class AgentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parallel: bool = True
    roles: dict[RoleName, RoleConfig]

    @model_validator(mode="after")
    def ensure_required_roles(self) -> "AgentsConfig":
        required: set[RoleName] = {"diagnostic", "treatment", "safety", "monitoring", "coordinator"}
        missing = required.difference(self.roles)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Missing agent role configuration: {missing_list}")
        return self


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enable_llm_judge: bool = False
    enable_bertscore: bool = False
    judge_llm: str | None = None


class EvolutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_weight_delta: float = 0.15
    weight_step: float = 0.05
    max_history: int = 200


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_registry: dict[str, LLMConfig]
    runtime: RuntimeConfig
    agents: AgentsConfig
    evaluation: EvaluationConfig
    evolution: EvolutionConfig
    logging: LoggingConfig

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        config_path = Path(path)
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        return cls(
            llm_registry=raw.get("llm", {}),
            runtime=raw.get("runtime", {}),
            agents=raw.get("agents", {}),
            evaluation=raw.get("evaluation", {}),
            evolution=raw.get("evolution", {}),
            logging=raw.get("logging", {}),
        )

    def llm_for(self, key: str) -> LLMConfig:
        if key not in self.llm_registry:
            raise KeyError(f"Unknown LLM config key: {key}")
        return self.llm_registry[key]

    @property
    def artifacts_dir(self) -> Path:
        return Path(self.runtime.artifacts_dir)

    @property
    def evolution_state_path(self) -> Path:
        return Path(self.runtime.evolution_state_path)


def configure_logging(config: AppConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, config.logging.level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
