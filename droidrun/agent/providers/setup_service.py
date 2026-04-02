from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from droidrun.agent.providers import (
    ProviderFamilySpec,
    ProviderVariantSpec,
    list_provider_families,
    resolve_provider_variant,
)
from droidrun.config_manager.config_manager import DroidConfig, LLMProfile
from droidrun.config_manager.env_keys import load_env_keys, save_env_keys


ENV_KEY_SLOTS_BY_VARIANT: dict[str, str] = {
    "GoogleGenAI": "google",
    "OpenAI": "openai",
    "Anthropic": "anthropic",
}

DEFAULT_KWARGS_BY_VARIANT: dict[str, dict[str, int]] = {
    "anthropic_oauth": {"max_tokens": 1024},
    "gemini_oauth_code_assist": {"max_tokens": 1024},
}

HIDDEN_ROLE_FALLBACKS: tuple[str, ...] = ("app_opener", "structured_output")


@dataclass(frozen=True)
class SetupSelection:
    family_id: str
    variant_id: str
    auth_mode: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    credential_path: str | None = None


def family_choices() -> tuple[ProviderFamilySpec, ...]:
    return list_provider_families()


def auth_mode_choices(family_id: str) -> tuple[str, ...]:
    family = next(f for f in list_provider_families() if f.id == family_id)
    return tuple(variant.auth_mode for variant in family.variants)


def variant_models(family_id: str, auth_mode: str) -> tuple[str, ...]:
    variant = resolve_provider_variant(family_id, auth_mode)
    return tuple(model.id for model in variant.models)


def create_profile_for_variant(
    variant: ProviderVariantSpec,
    selection: SetupSelection,
    *,
    temperature: float = 0.2,
) -> LLMProfile:
    base_url = selection.base_url or variant.base_url
    kwargs: dict[str, str | int] = dict(DEFAULT_KWARGS_BY_VARIANT.get(variant.id, {}))
    env_slot = ENV_KEY_SLOTS_BY_VARIANT.get(variant.id)
    runtime_provider_name = (
        variant.runtime_transport_provider_name or variant.runtime_provider_name
    )

    if variant.id == "OpenAILike":
        kwargs["api_key"] = selection.api_key or "stub"
    elif variant.id == "ZAI":
        kwargs["api_key"] = selection.api_key or "stub"
    elif variant.id == "ZAI_Coding":
        kwargs["api_key"] = selection.api_key or "stub"
    elif variant.id == "MiniMax":
        kwargs["api_key"] = selection.api_key or ""

    return LLMProfile(
        provider=runtime_provider_name,
        provider_family=selection.family_id,
        auth_mode=selection.auth_mode,
        model=selection.model,
        temperature=temperature,
        base_url=base_url,
        api_base=base_url if runtime_provider_name == "OpenAILike" else None,
        credential_path=selection.credential_path or variant.credential_path,
        kwargs=kwargs if env_slot is None else {},
    )


def apply_selection_to_roles(
    config: DroidConfig,
    selection: SetupSelection,
    roles: Iterable[str],
) -> DroidConfig:
    variant = resolve_provider_variant(selection.family_id, selection.auth_mode)
    env_slot = ENV_KEY_SLOTS_BY_VARIANT.get(variant.id)
    if selection.api_key and env_slot:
        existing = load_env_keys()
        existing[env_slot] = selection.api_key
        try:
            save_env_keys(existing)
        except OSError:
            pass

    if variant.id == "anthropic_oauth":
        config.agent.streaming = False

    for role in roles:
        if role not in config.llm_profiles:
            continue
        current = config.llm_profiles[role]
        config.llm_profiles[role] = create_profile_for_variant(
            variant,
            selection,
            temperature=current.temperature,
        )

    if "fast_agent" in roles:
        fast_agent_profile = config.llm_profiles.get("fast_agent")
        if fast_agent_profile is not None:
            for hidden_role in HIDDEN_ROLE_FALLBACKS:
                if hidden_role not in config.llm_profiles:
                    continue
                current = config.llm_profiles[hidden_role]
                config.llm_profiles[hidden_role] = LLMProfile(
                    provider=fast_agent_profile.provider,
                    model=fast_agent_profile.model,
                    temperature=current.temperature,
                    base_url=fast_agent_profile.base_url,
                    api_base=fast_agent_profile.api_base,
                    provider_family=fast_agent_profile.provider_family,
                    auth_mode=fast_agent_profile.auth_mode,
                    credential_path=fast_agent_profile.credential_path,
                    kwargs=dict(fast_agent_profile.kwargs),
                )

    return config
