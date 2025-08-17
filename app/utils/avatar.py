import random
import uuid

from ..core.config import settings


def select_random_avatar() -> str:
    style = random.choice(settings.AVATAR_STYLES)
    seed = str(uuid.uuid4())
    avatar_url = f"{settings.AVATAR_PROVIDER}/{style}/svg?seed={seed}"

    return avatar_url
