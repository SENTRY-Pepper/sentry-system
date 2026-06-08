from typing import Callable

from fastapi import Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import get_db
from backend.database.models import User


async def get_current_user(
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")

    token = authorization.split(" ", 1)[1].strip()
    user_id = token.split(".", 1)[0]
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid authentication token.")

    user = await db.get(User, user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Invalid authentication token.")
    return user


def require_roles(*roles: str) -> Callable:
    allowed = {role.lower() for role in roles}

    async def dependency(user: User = Depends(get_current_user)) -> User:
        if user.role.lower() not in allowed:
            raise HTTPException(status_code=403, detail="Insufficient role.")
        return user

    return dependency
