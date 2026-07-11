"""Remote provider adapters for API-first hill climbs."""

from post_train_engine.providers.base import RemoteProvider
from post_train_engine.providers.fake import FakeInferenceProvider, FakePromptAdapterProvider
from post_train_engine.providers.openai_compatible import OpenAICompatibleProvider

__all__ = [
    "FakeInferenceProvider",
    "FakePromptAdapterProvider",
    "OpenAICompatibleProvider",
    "RemoteProvider",
]
