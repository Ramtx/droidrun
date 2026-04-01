from droidrun.agent.providers import (
    get_provider_family,
    list_auth_modes,
    list_models_for_variant,
    resolve_provider_variant,
)
from droidrun.config_manager import config_manager
from droidrun.config_manager.config_manager import LLMProfile


def test_openai_family_exposes_api_and_oauth_variants() -> None:
    assert list_auth_modes("openai") == ("api_key", "oauth")


def test_openai_oauth_model_catalog_is_restricted() -> None:
    model_ids = [model.id for model in list_models_for_variant("openai", "oauth")]
    assert model_ids == [
        "gpt-5.4",
        "gpt-5.3-codex",
    ]


def test_zai_is_first_class_but_uses_openai_like_transport() -> None:
    variant = resolve_provider_variant("zai", "api_key")
    assert variant.id == "ZAI"
    assert variant.runtime_provider_name == "ZAI"
    assert variant.runtime_transport_provider_name == "OpenAILike"
    assert variant.base_url == "https://api.z.ai/api/paas/v4"


def test_zai_exposes_dedicated_coding_api_variant() -> None:
    assert list_auth_modes("zai") == ("api_key", "coding_api")

    variant = resolve_provider_variant("zai", "coding_api")

    assert variant.id == "ZAI_Coding"
    assert variant.runtime_provider_name == "ZAI"
    assert variant.runtime_transport_provider_name == "OpenAILike"
    assert variant.base_url == "https://api.z.ai/api/coding/paas/v4"


def test_minimax_stays_first_class() -> None:
    variant = resolve_provider_variant("minimax", "api_key")
    assert variant.runtime_provider_name == "MiniMax"
    assert variant.runtime_transport_provider_name is None


def test_family_lookup_uses_user_facing_id() -> None:
    family = get_provider_family("anthropic")
    assert family.display_name == "Anthropic"
    assert tuple(variant.id for variant in family.variants) == (
        "Anthropic",
        "anthropic_oauth",
    )


def test_llm_profile_framework_metadata_is_backward_compatible() -> None:
    profile = LLMProfile(provider="OpenAI", model="gpt-5.4")
    assert profile.provider_family is None
    assert profile.auth_mode is None
    assert profile.credential_path is None
    kwargs = profile.to_load_llm_kwargs()
    assert kwargs["model"] == "gpt-5.4"
    assert "credential_path" not in kwargs


def test_llm_profile_injects_env_api_key_for_google_genai(monkeypatch) -> None:
    monkeypatch.setattr(
        config_manager,
        "load_env_keys",
        lambda: {"google": "test-google-key", "openai": "", "anthropic": ""},
    )
    profile = LLMProfile(provider="GoogleGenAI", model="gemini-2.5-flash")

    kwargs = profile.to_load_llm_kwargs()

    assert kwargs["api_key"] == "test-google-key"
