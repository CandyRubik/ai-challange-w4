from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class StoredMessage:
    id: str
    role: str
    kind: str
    content: str
    created_at: datetime
    refusal: bool = False
    sources: tuple[dict[str, Any], ...] = ()
