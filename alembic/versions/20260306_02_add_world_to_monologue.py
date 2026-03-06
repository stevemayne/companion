"""add world jsonb column to monologue_states

Stores the companion's subjective world model (CharacterState for self,
user, other characters, environment, recent events).

Revision ID: 20260306_02
Revises: 20260306_01
Create Date: 2026-03-06
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "20260306_02"
down_revision = "20260306_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("monologue_states")}
    if "world" not in columns:
        op.add_column(
            "monologue_states",
            sa.Column(
                "world",
                postgresql.JSONB(),
                nullable=False,
                server_default=sa.text("'{}'::jsonb"),
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("monologue_states")}
    if "world" in columns:
        op.drop_column("monologue_states", "world")
