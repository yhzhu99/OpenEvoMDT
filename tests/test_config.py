from evomdt.config import AppConfig


def test_load_config_contains_required_roles():
    config = AppConfig.load("config.toml")
    assert config.runtime.default_llm == "deepseek/deepseek-chat"
    assert set(config.agents.roles) == {"diagnostic", "treatment", "safety", "monitoring", "coordinator"}
