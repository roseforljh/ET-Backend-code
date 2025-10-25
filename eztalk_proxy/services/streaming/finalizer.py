"""
Finalization helpers extracted from legacy stream_processor.
This module is now a no-op placeholder after format repair was disabled.
"""

from typing import Any, Tuple

def finalize_delta(
    delta_text: str,
    request_data: Any | None,
    log_prefix: str,
    allow_final_repair: bool = True,
) -> Tuple[str, str]:
    """
    No-op finalizer, returns original text and 'general' type.
    """
    return delta_text, "general"