"""Shared messaging dispatch for pykoclaw channel plugins."""

from pykoclaw_messaging.dispatch import DispatchResult, dispatch_to_agent
from pykoclaw_messaging.plugin import MessagingPlugin

__all__ = ["DispatchResult", "MessagingPlugin", "dispatch_to_agent"]
