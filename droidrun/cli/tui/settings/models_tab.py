"""Models tab — per-agent LLM profile cards with apply-to-all."""

from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.containers import HorizontalGroup, VerticalGroup
from textual.widgets import Button, Input, Label, Select, Static
from textual import on

from droidrun.cli.oauth_actions import (
    run_anthropic_oauth_setup,
    run_gemini_oauth_login,
    run_openai_oauth_login,
)
from droidrun.cli.tui.settings.data import (
    AGENT_ROLES,
    PROVIDERS,
    ProfileSettings,
    SettingsData,
    provider_fields,
    provider_label,
    provider_credential_path,
    provider_models,
    provider_oauth_command,
    provider_oauth_status,
)
from droidrun.cli.tui.settings.section import Section


PROVIDER_OPTIONS = [(provider_label(p), p) for p in PROVIDERS]


class _KwargsRow(HorizontalGroup):
    """A single key-value pair row with a remove button."""

    CSS_PATH = "../css/models_tab.tcss"

    def __init__(self, key: str, value: str, row_id: str) -> None:
        super().__init__()
        self._key = key
        self._value = value
        self._row_id = row_id

    def compose(self) -> ComposeResult:
        yield Input(
            value=self._key,
            placeholder="key",
            classes="kwarg-key",
            id=f"kk-{self._row_id}",
        )
        yield Input(value=self._value, placeholder="value", id=f"kv-{self._row_id}")
        yield Button("×", id=f"kr-{self._row_id}")


class _KwargsEditor(VerticalGroup):
    """Editable key-value pair list."""

    CSS_PATH = "../css/models_tab.tcss"

    def __init__(self, kwargs: dict[str, str], role: str) -> None:
        super().__init__()
        self._kwargs = kwargs
        self._role = role
        self._counter = 0

    def compose(self) -> ComposeResult:
        for key, value in self._kwargs.items():
            rid = f"{self._role}-{self._counter}"
            self._counter += 1
            yield _KwargsRow(key, value, rid)
        yield Button("+ add", classes="kwargs-add-btn", id=f"kwargs-add-{self._role}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id == f"kwargs-add-{self._role}":
            event.stop()
            rid = f"{self._role}-{self._counter}"
            self._counter += 1
            self.mount(_KwargsRow("", "", rid), before=event.button)
        elif btn_id.startswith("kr-"):
            event.stop()
            row = event.button.parent
            if row:
                row.remove()

    def collect(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for row in self.query(_KwargsRow):
            key_input = row.query("Input.kwarg-key")
            val_input = row.query("Input:not(.kwarg-key)")
            if key_input and val_input:
                k = key_input.first().value.strip()
                v = val_input.first().value.strip()
                if k:
                    result[k] = v
        return result


class _ProfileCard(Section):
    """Full LLM config card for one agent role."""

    CSS_PATH = "../css/models_tab.tcss"

    def __init__(self, role: str, profile: ProfileSettings) -> None:
        super().__init__(title=role.title())
        self._role = role
        self._profile = profile

    def compose(self) -> ComposeResult:
        if self._role == "manager":
            yield Button("Apply to all", id="apply-all-btn")

        model_options = [(m, m) for m in provider_models(self._profile.provider)]
        has_model_choices = bool(model_options)
        selected_model = (
            self._profile.model
            if self._profile.model in provider_models(self._profile.provider)
            else (provider_models(self._profile.provider)[0] if has_model_choices else Select.BLANK)
        )
        pf = provider_fields(self._profile.provider)

        with VerticalGroup(classes="profile-fields"):
            with HorizontalGroup(classes="field-row"):
                yield Label("Provider", classes="field-label")
                yield Select(
                    PROVIDER_OPTIONS,
                    value=self._profile.provider,
                    allow_blank=False,
                    id=f"provider-{self._role}",
                    classes="field-select",
                )

            with HorizontalGroup(classes="field-row"):
                yield Label("Model", classes="field-label")
                yield Select(
                    model_options or [("Custom", Select.BLANK)],
                    value=selected_model,
                    allow_blank=not has_model_choices,
                    id=f"model-select-{self._role}",
                    classes="field-select" if has_model_choices else "field-select hidden-field",
                )
                yield Input(
                    value=self._profile.model,
                    id=f"model-input-{self._role}",
                    classes="field-input" if not has_model_choices else "field-input hidden-field",
                )

            api_key_cls = "field-row" if pf.get("api_key") else "field-row hidden-field"
            with HorizontalGroup(classes=api_key_cls, id=f"row-apikey-{self._role}"):
                yield Label("API Key", classes="field-label")
                yield Input(
                    value=self._profile.api_key,
                    password=True,
                    id=f"apikey-{self._role}",
                    classes="field-input",
                )

            url_cls = "field-row" if pf.get("base_url") else "field-row hidden-field"
            with HorizontalGroup(classes=url_cls, id=f"row-baseurl-{self._role}"):
                yield Label("Base URL", classes="field-label")
                yield Input(
                    value=self._profile.base_url,
                    placeholder="http://localhost:11434",
                    id=f"baseurl-{self._role}",
                    classes="field-input",
                )

            with HorizontalGroup(classes="field-row"):
                yield Label("Temperature", classes="field-label")
                yield Input(
                    value=str(self._profile.temperature),
                    id=f"temp-{self._role}",
                    classes="field-input",
                )

            with HorizontalGroup(classes="field-row"):
                yield Label("Max Tokens", classes="field-label")
                yield Input(
                    value=self._profile.max_tokens,
                    id=f"max-tokens-{self._role}",
                    classes="field-input",
                )

            oauth_command = provider_oauth_command(self._profile.provider)
            oauth_status = provider_oauth_status(
                self._profile.provider, self._profile.credential_path
            )
            oauth_cls = (
                "field-row" if oauth_command is not None else "field-row hidden-field"
            )
            with HorizontalGroup(classes=oauth_cls, id=f"row-oauth-{self._role}"):
                yield Label("OAuth", classes="field-label")
                yield Static(
                    oauth_status or "Not configured",
                    id=f"oauth-status-{self._role}",
                    classes="oauth-status",
                )
                yield Button("Open auth flow", id=f"oauth-login-{self._role}")

        yield Static("extra parameters", classes="kwargs-label")
        yield _KwargsEditor(self._profile.kwargs, self._role)

    @on(Select.Changed)
    def _on_provider_changed(self, event: Select.Changed) -> None:
        if event.select.id != f"provider-{self._role}":
            return
        provider = str(event.value)
        pf = provider_fields(provider)

        api_row = self.query_one(f"#row-apikey-{self._role}")
        url_row = self.query_one(f"#row-baseurl-{self._role}")
        oauth_row = self.query_one(f"#row-oauth-{self._role}")
        oauth_status = self.query_one(f"#oauth-status-{self._role}", Static)
        model_select = self.query_one(f"#model-select-{self._role}", Select)
        model_input = self.query_one(f"#model-input-{self._role}", Input)
        model_choices = provider_models(provider)
        current_path = (
            self._profile.credential_path
            if provider == self._profile.provider and self._profile.credential_path
            else provider_credential_path(provider)
        )
        oauth_message = provider_oauth_status(provider, current_path)

        if pf.get("api_key"):
            api_row.remove_class("hidden-field")
        else:
            api_row.add_class("hidden-field")

        if pf.get("base_url"):
            url_row.remove_class("hidden-field")
        else:
            url_row.add_class("hidden-field")

        if provider_oauth_command(provider):
            oauth_row.remove_class("hidden-field")
            oauth_status.update(oauth_message or "Not configured")
        else:
            oauth_row.add_class("hidden-field")

        if model_choices:
            model_select.set_options([(m, m) for m in model_choices])
            model_select.value = model_choices[0]
            model_select.remove_class("hidden-field")
            model_input.add_class("hidden-field")
            model_input.value = model_choices[0]
        else:
            model_select.add_class("hidden-field")
            model_input.remove_class("hidden-field")
            if not model_input.value.strip():
                model_input.value = ""

    def collect(self) -> ProfileSettings:
        provider = str(self.query_one(f"#provider-{self._role}", Select).value)
        model_choices = provider_models(provider)
        if model_choices:
            model = str(self.query_one(f"#model-select-{self._role}", Select).value)
        else:
            model = self.query_one(f"#model-input-{self._role}", Input).value.strip()
        api_key = self.query_one(f"#apikey-{self._role}", Input).value.strip()
        base_url = self.query_one(f"#baseurl-{self._role}", Input).value.strip()
        temp_str = self.query_one(f"#temp-{self._role}", Input).value.strip()
        max_tokens = self.query_one(f"#max-tokens-{self._role}", Input).value.strip()
        try:
            temperature = float(temp_str)
        except (ValueError, TypeError):
            temperature = 0.2
        kwargs = self.query_one(_KwargsEditor).collect()
        return ProfileSettings(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            credential_path=(
                self._profile.credential_path
                if provider == self._profile.provider and self._profile.credential_path
                else (provider_credential_path(provider) or "")
            ),
            kwargs=kwargs,
        )


class ModelsTab(VerticalGroup):
    """Content for the Models tab pane — per-agent profile cards."""

    CSS_PATH = "../css/models_tab.tcss"

    def __init__(self, settings: SettingsData) -> None:
        super().__init__()
        self.settings = settings

    def compose(self) -> ComposeResult:
        for role in AGENT_ROLES:
            profile = self.settings.profiles.get(role, ProfileSettings())
            yield _ProfileCard(role, profile)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply-all-btn":
            event.stop()
            self._apply_first_to_all()
        elif event.button.id and event.button.id.startswith("oauth-login-"):
            event.stop()
            role = event.button.id.removeprefix("oauth-login-")
            card = next(card for card in self.query(_ProfileCard) if card._role == role)
            provider = str(card.query_one(f"#provider-{role}", Select).value)
            credential_path = provider_credential_path(provider)
            model_choices = provider_models(provider)
            if model_choices:
                model = str(card.query_one(f"#model-select-{role}", Select).value)
            else:
                model = card.query_one(f"#model-input-{role}", Input).value.strip()
            self.run_worker(
                self._run_oauth_flow(role, provider, credential_path, model),
                group=f"oauth-{role}",
                exclusive=True,
            )

    async def _run_oauth_flow(
        self, role: str, provider: str, credential_path: str | None, model: str
    ) -> None:
        status = next(
            card.query_one(f"#oauth-status-{role}", Static)
            for card in self.query(_ProfileCard)
            if card._role == role
        )
        status.update("Opening browser...")
        try:
            await asyncio.to_thread(
                self._run_oauth_flow_blocking, provider, credential_path, model
            )
        except Exception as e:
            status.update("Auth failed")
            command = provider_oauth_command(provider)
            message = str(e)
            if command:
                message = f"{message}. Use: {command}"
            self.app.notify(message, title="OAuth Error", timeout=6)
            return

        updated = provider_oauth_status(provider, credential_path) or "Configured"
        status.update(updated)
        card = next(card for card in self.query(_ProfileCard) if card._role == role)
        card._profile.credential_path = credential_path or card._profile.credential_path
        card._profile.provider = provider
        card._profile.model = model
        self.app.notify(f"{provider_label(provider)} ready", title="OAuth", timeout=3)

    def _run_oauth_flow_blocking(
        self, provider: str, credential_path: str | None, model: str
    ) -> None:
        if not credential_path:
            raise RuntimeError("This provider does not use OAuth credentials.")

        if provider == "anthropic_oauth":
            run_anthropic_oauth_setup(credential_path)
        elif provider == "openai_oauth":
            run_openai_oauth_login(credential_path=credential_path, model=model)
        elif provider == "gemini_oauth_code_assist":
            run_gemini_oauth_login(credential_path=credential_path, model=model)
        else:
            raise RuntimeError("Unsupported OAuth provider.")

    def _apply_first_to_all(self) -> None:
        """Copy the first profile card's values to all other cards."""
        cards = list(self.query(_ProfileCard))
        if len(cards) < 2:
            return
        source = cards[0].collect()
        for card in cards[1:]:
            card.query_one(f"#provider-{card._role}", Select).value = source.provider
            card.query_one(f"#apikey-{card._role}", Input).value = source.api_key
            card.query_one(f"#baseurl-{card._role}", Input).value = source.base_url
            card.query_one(f"#temp-{card._role}", Input).value = str(source.temperature)
            card.query_one(f"#max-tokens-{card._role}", Input).value = source.max_tokens
            # Trigger field visibility update
            pf = provider_fields(source.provider)
            api_row = card.query_one(f"#row-apikey-{card._role}")
            url_row = card.query_one(f"#row-baseurl-{card._role}")
            oauth_row = card.query_one(f"#row-oauth-{card._role}")
            oauth_status = card.query_one(f"#oauth-status-{card._role}", Static)
            model_select = card.query_one(f"#model-select-{card._role}", Select)
            model_input = card.query_one(f"#model-input-{card._role}", Input)
            model_choices = provider_models(source.provider)
            if pf.get("api_key"):
                api_row.remove_class("hidden-field")
            else:
                api_row.add_class("hidden-field")
            if pf.get("base_url"):
                url_row.remove_class("hidden-field")
            else:
                url_row.add_class("hidden-field")
            if provider_oauth_command(source.provider):
                oauth_row.remove_class("hidden-field")
                oauth_status.update(
                    provider_oauth_status(
                        source.provider, source.credential_path or provider_credential_path(source.provider)
                    )
                    or "Not configured"
                )
            else:
                oauth_row.add_class("hidden-field")
            if model_choices:
                model_select.set_options([(m, m) for m in model_choices])
                model_select.value = (
                    source.model if source.model in model_choices else model_choices[0]
                )
                model_select.remove_class("hidden-field")
                model_input.add_class("hidden-field")
                model_input.value = model_select.value
            else:
                model_select.add_class("hidden-field")
                model_input.remove_class("hidden-field")
                model_input.value = source.model

    def collect(self) -> dict[str, ProfileSettings]:
        result: dict[str, ProfileSettings] = {}
        for card in self.query(_ProfileCard):
            result[card._role] = card.collect()
        return result
