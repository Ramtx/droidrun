"""Settings data model for the TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from droidrun.config_manager.config_manager import DroidConfig

from droidrun.agent.providers.registry import list_provider_families, resolve_provider_variant
from droidrun.agent.providers.setup_service import (
    SetupSelection,
    create_profile_for_variant,
)
from droidrun.config_manager.env_keys import load_env_keys, save_env_keys
from droidrun.config_manager.credential_paths import (
    ANTHROPIC_OAUTH_CREDENTIAL_PATH,
    GEMINI_OAUTH_CREDENTIAL_PATH,
    OPENAI_OAUTH_CREDENTIAL_PATH,
)

AGENT_ROLES = ["manager", "executor", "fast_agent"]


@dataclass(frozen=True)
class ProviderChoice:
    id: str
    label: str
    family_id: str
    auth_mode: str
    runtime_provider_name: str
    requires_api_key: bool
    requires_base_url: bool
    credential_path: str | None
    base_url: str | None
    models: tuple[str, ...]


def _provider_label(display_name: str, auth_mode: str) -> str:
    if auth_mode == "none":
        return display_name
    if auth_mode == "api_key":
        return f"{display_name} (API key)"
    if auth_mode == "oauth":
        return f"{display_name} (OAuth)"
    return f"{display_name} ({auth_mode.replace('_', ' ')})"


def _build_provider_choices() -> dict[str, ProviderChoice]:
    result: dict[str, ProviderChoice] = {}
    for family in list_provider_families():
        for variant in family.variants:
            result[variant.id] = ProviderChoice(
                id=variant.id,
                label=_provider_label(family.display_name, variant.auth_mode),
                family_id=family.id,
                auth_mode=variant.auth_mode,
                runtime_provider_name=variant.runtime_provider_name,
                requires_api_key=variant.requires_api_key,
                requires_base_url=variant.requires_base_url,
                credential_path=variant.credential_path,
                base_url=variant.base_url,
                models=tuple(model.id for model in variant.models),
            )
    return result


PROVIDER_CHOICES = _build_provider_choices()
PROVIDERS = list(PROVIDER_CHOICES.keys())

# Maps provider variant id to the env key slot used by save_env_keys/load_env_keys.
PROVIDER_ENV_KEY_SLOT: dict[str, str] = {
    "GoogleGenAI": "google",
    "OpenAI": "openai",
    "Anthropic": "anthropic",
}


def provider_fields(variant_id: str) -> dict[str, bool]:
    choice = PROVIDER_CHOICES[variant_id]
    return {
        "api_key": choice.requires_api_key,
        "base_url": choice.requires_base_url,
    }


def provider_label(variant_id: str) -> str:
    return PROVIDER_CHOICES[variant_id].label


def provider_family_id(variant_id: str) -> str:
    return PROVIDER_CHOICES[variant_id].family_id


def provider_auth_mode(variant_id: str) -> str:
    return PROVIDER_CHOICES[variant_id].auth_mode


def provider_family_label(family_id: str) -> str:
    for family in list_provider_families():
        if family.id == family_id:
            return family.display_name
    return family_id


def provider_family_options() -> list[tuple[str, str]]:
    return [(family.display_name, family.id) for family in list_provider_families()]


def provider_auth_mode_options(family_id: str) -> list[tuple[str, str]]:
    family = next(f for f in list_provider_families() if f.id == family_id)
    return [
        (variant.auth_mode.replace("_", " "), variant.auth_mode)
        for variant in family.variants
    ]


def resolve_variant_id(family_id: str, auth_mode: str) -> str:
    return resolve_provider_variant(family_id, auth_mode).id


def provider_models(variant_id: str) -> tuple[str, ...]:
    return PROVIDER_CHOICES[variant_id].models


def provider_credential_path(variant_id: str) -> str | None:
    return PROVIDER_CHOICES[variant_id].credential_path


def provider_oauth_command(variant_id: str) -> str | None:
    if variant_id == "anthropic_oauth":
        return "droidrun setup-token"
    if variant_id == "openai_oauth":
        return "droidrun openai login"
    if variant_id == "gemini_oauth_code_assist":
        return "droidrun gemini login"
    return None


def provider_oauth_status(variant_id: str, credential_path: str | None = None) -> str | None:
    path = Path(credential_path).expanduser() if credential_path else None
    if path is None:
        if variant_id == "anthropic_oauth":
            path = ANTHROPIC_OAUTH_CREDENTIAL_PATH
        elif variant_id == "openai_oauth":
            path = OPENAI_OAUTH_CREDENTIAL_PATH
        elif variant_id == "gemini_oauth_code_assist":
            path = GEMINI_OAUTH_CREDENTIAL_PATH
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "Credential file found"
    if variant_id == "anthropic_oauth":
        token = (payload.get("claudeAiOauth") or {}).get("accessToken")
        return "Token saved" if token else "Credential file found"
    if variant_id == "openai_oauth":
        token = payload.get("access") or payload.get("access_token")
        return "Logged in" if token else "Credential file found"
    if variant_id == "gemini_oauth_code_assist":
        token = payload.get("access_token")
        return "Logged in" if token else "Credential file found"
    return None


def resolve_variant_id_from_profile(profile: Any) -> str:
    family_id = getattr(profile, "provider_family", None)
    auth_mode = getattr(profile, "auth_mode", None)
    if family_id and auth_mode:
        try:
            return resolve_provider_variant(family_id, auth_mode).id
        except Exception:
            pass

    provider_name = getattr(profile, "provider", "")
    for choice in PROVIDER_CHOICES.values():
        if choice.runtime_provider_name == provider_name:
            return choice.id
    return "GoogleGenAI"


@dataclass
class ProfileSettings:
    """Full LLM profile for one agent role."""

    provider: str = "GoogleGenAI"
    model: str = "gemini-3.1-flash-lite-preview"
    temperature: float = 0.2
    max_tokens: str = ""
    api_key: str = ""
    base_url: str = ""
    credential_path: str = ""
    kwargs: dict[str, str] = field(default_factory=dict)


@dataclass
class SettingsData:
    """All TUI settings in one object."""

    profiles: dict[str, ProfileSettings] = field(
        default_factory=lambda: {role: ProfileSettings() for role in AGENT_ROLES}
    )
    agent_prompts: dict[str, str] = field(
        default_factory=lambda: {role: "" for role in AGENT_ROLES}
    )
    manager_vision: bool = True
    executor_vision: bool = False
    fast_agent_vision: bool = False
    reasoning: bool = False
    manager_stateless: bool = False
    max_steps: int = 15
    use_tcp: bool = False
    debug: bool = False
    save_trajectory: bool = False
    trajectory_gifs: bool = True
    tracing_enabled: bool = False
    tracing_provider: str = "phoenix"
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_screenshots: bool = False
    after_sleep_action: float = 1.0
    wait_for_stable_ui: float = 0.3

    @classmethod
    def from_config(cls, config: DroidConfig) -> SettingsData:
        llm_profiles = config.llm_profiles or {}
        env_keys = load_env_keys()

        profiles: dict[str, ProfileSettings] = {}
        for role in AGENT_ROLES:
            lp = llm_profiles.get(role)
            if lp:
                variant_id = resolve_variant_id_from_profile(lp)
                env_slot = PROVIDER_ENV_KEY_SLOT.get(variant_id)
                if env_slot:
                    api_key = env_keys.get(env_slot, "")
                elif variant_id in {"OpenAILike", "ZAI", "ZAI_Coding", "MiniMax"}:
                    api_key = str(lp.kwargs.get("api_key", "") or "")
                else:
                    api_key = ""

                kwargs = {k: str(v) for k, v in lp.kwargs.items() if k != "api_key"}
                max_tokens = str(lp.kwargs.get("max_tokens", "") or "")
                kwargs.pop("max_tokens", None)

                profiles[role] = ProfileSettings(
                    provider=variant_id,
                    model=lp.model,
                    temperature=lp.temperature,
                    max_tokens=max_tokens,
                    api_key=api_key,
                    base_url=lp.base_url or lp.api_base or "",
                    credential_path=lp.credential_path or "",
                    kwargs=kwargs,
                )
            else:
                profiles[role] = ProfileSettings()

        agent_prompts = {
            "manager": config.agent.manager.system_prompt,
            "executor": config.agent.executor.system_prompt,
            "fast_agent": config.agent.fast_agent.system_prompt,
        }

        return cls(
            profiles=profiles,
            agent_prompts=agent_prompts,
            manager_vision=config.agent.manager.vision,
            executor_vision=config.agent.executor.vision,
            fast_agent_vision=config.agent.fast_agent.vision,
            reasoning=config.agent.reasoning,
            manager_stateless=config.agent.manager.stateless,
            max_steps=config.agent.max_steps,
            use_tcp=config.device.use_tcp,
            debug=config.logging.debug,
            save_trajectory=config.logging.save_trajectory != "none",
            trajectory_gifs=config.logging.trajectory_gifs,
            tracing_enabled=config.tracing.enabled,
            tracing_provider=config.tracing.provider,
            langfuse_host=config.tracing.langfuse_host,
            langfuse_public_key=config.tracing.langfuse_public_key,
            langfuse_secret_key=config.tracing.langfuse_secret_key,
            langfuse_screenshots=config.tracing.langfuse_screenshots,
            after_sleep_action=config.agent.after_sleep_action,
            wait_for_stable_ui=config.agent.wait_for_stable_ui,
        )

    def save(self) -> None:
        from droidrun.config_manager.loader import ConfigLoader

        env_keys = load_env_keys()
        for _, profile in self.profiles.items():
            env_slot = PROVIDER_ENV_KEY_SLOT.get(profile.provider)
            if env_slot:
                env_keys[env_slot] = profile.api_key.strip()
        save_env_keys(env_keys)

        try:
            config = ConfigLoader.load()
        except Exception:
            from droidrun.config_manager.config_manager import DroidConfig

            config = DroidConfig()

        self.apply_to_config(config)
        ConfigLoader.save(config)

    @staticmethod
    def _build_kwargs(ps: ProfileSettings) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for k, v in ps.kwargs.items():
            if not k:
                continue
            try:
                parsed[k] = int(v)
            except ValueError:
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = v

        if ps.provider in {"OpenAILike", "ZAI", "ZAI_Coding", "MiniMax"}:
            parsed["api_key"] = ps.api_key or ("stub" if ps.provider != "MiniMax" else "")
        if ps.max_tokens.strip():
            try:
                parsed["max_tokens"] = int(ps.max_tokens.strip())
            except ValueError:
                parsed["max_tokens"] = ps.max_tokens.strip()
        return parsed

    @staticmethod
    def _apply_profile_to_llm(
        ps: ProfileSettings, cp: Any, update_model: bool = True
    ) -> None:
        choice = PROVIDER_CHOICES[ps.provider]
        selection = SetupSelection(
            family_id=choice.family_id,
            variant_id=choice.id,
            auth_mode=choice.auth_mode,
            model=ps.model,
            api_key=ps.api_key or None,
            base_url=ps.base_url or None,
            credential_path=ps.credential_path or choice.credential_path,
        )
        variant = resolve_provider_variant(choice.family_id, choice.auth_mode)
        generated = create_profile_for_variant(
            variant,
            selection,
            temperature=ps.temperature,
        )

        cp.provider = generated.provider
        cp.provider_family = generated.provider_family
        cp.auth_mode = generated.auth_mode
        cp.credential_path = generated.credential_path
        cp.base_url = generated.base_url
        cp.api_base = generated.api_base
        if update_model:
            cp.model = generated.model
        cp.temperature = ps.temperature
        cp.kwargs = generated.kwargs | SettingsData._build_kwargs(ps)

    def apply_to_config(self, config: DroidConfig) -> None:
        for role, ps in self.profiles.items():
            if role not in config.llm_profiles:
                continue
            self._apply_profile_to_llm(ps, config.llm_profiles[role])

        prompt = self.agent_prompts.get("manager", "")
        if prompt:
            config.agent.manager.system_prompt = prompt
        prompt = self.agent_prompts.get("executor", "")
        if prompt:
            config.agent.executor.system_prompt = prompt
        prompt = self.agent_prompts.get("fast_agent", "")
        if prompt:
            config.agent.fast_agent.system_prompt = prompt

        config.agent.max_steps = self.max_steps
        config.agent.reasoning = self.reasoning
        config.agent.manager.stateless = self.manager_stateless
        config.agent.manager.vision = self.manager_vision
        config.agent.executor.vision = self.executor_vision
        config.agent.fast_agent.vision = self.fast_agent_vision
        config.device.use_tcp = self.use_tcp
        config.logging.debug = self.debug
        config.logging.save_trajectory = "action" if self.save_trajectory else "none"
        config.logging.trajectory_gifs = self.trajectory_gifs
        config.tracing.enabled = self.tracing_enabled
        config.tracing.provider = self.tracing_provider
        config.tracing.langfuse_host = self.langfuse_host
        config.tracing.langfuse_public_key = self.langfuse_public_key
        config.tracing.langfuse_secret_key = self.langfuse_secret_key
        config.tracing.langfuse_screenshots = self.langfuse_screenshots
        config.agent.after_sleep_action = self.after_sleep_action
        config.agent.wait_for_stable_ui = self.wait_for_stable_ui
