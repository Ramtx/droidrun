"""Agent tab — reasoning, grouped vision, max steps, prompts."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import HorizontalGroup, VerticalGroup
from textual.widgets import Input, Label

from droidrun.cli.tui.settings.data import AGENT_ROLES, SettingsData
from droidrun.cli.tui.settings.section import BoolToggle, Section


class AgentTab(VerticalGroup):
    """Content for the Agent tab pane."""

    def __init__(self, settings: SettingsData) -> None:
        super().__init__()
        self.settings = settings

    def compose(self) -> ComposeResult:
        with Section("Mode"):
            with HorizontalGroup(classes="field-row"):
                yield Label("Reasoning", classes="field-label")
                yield BoolToggle(value=self.settings.reasoning, id="reasoning-mode")

            with HorizontalGroup(classes="field-row"):
                yield Label("Manager Stateless", classes="field-label")
                yield BoolToggle(
                    value=self.settings.manager_stateless,
                    id="manager-stateless",
                )

        with Section("Vision", hint="send screenshots to the LLM"):
            with HorizontalGroup(classes="field-row"):
                yield Label("Planning Mode", classes="field-label")
                yield BoolToggle(
                    value=(
                        self.settings.manager_vision or self.settings.executor_vision
                    ),
                    id="vision-planning",
                )

            with HorizontalGroup(classes="field-row"):
                yield Label("Direct Mode", classes="field-label")
                yield BoolToggle(
                    value=self.settings.fast_agent_vision, id="vision-fast-agent"
                )

        with Section("Steps"):
            with HorizontalGroup(classes="field-row"):
                yield Label("Max Steps", classes="field-label")
                yield Input(
                    value=str(self.settings.max_steps),
                    id="max-steps",
                    classes="field-input",
                )

        with Section("Prompts", hint="system prompt path per agent"):
            for role in AGENT_ROLES:
                with HorizontalGroup(classes="field-row"):
                    yield Label(role.title(), classes="field-label")
                    yield Input(
                        value=self.settings.agent_prompts.get(role, ""),
                        id=f"prompt-{role}",
                        classes="field-input",
                    )

    def collect(self) -> dict:
        """Collect current agent settings."""
        return {
            "reasoning": self.query_one("#reasoning-mode", BoolToggle).value,
            "manager_stateless": self.query_one(
                "#manager-stateless", BoolToggle
            ).value,
            "manager_vision": self.query_one("#vision-planning", BoolToggle).value,
            "executor_vision": self.query_one("#vision-planning", BoolToggle).value,
            "fast_agent_vision": self.query_one("#vision-fast-agent", BoolToggle).value,
            "max_steps": self.query_one("#max-steps", Input).value.strip(),
            "agent_prompts": {
                role: self.query_one(f"#prompt-{role}", Input).value.strip()
                for role in AGENT_ROLES
            },
        }
