"""add user_state column to monologue_states

Revision ID: 20260302_03
Revises: 20260302_02
Create Date: 2026-03-02 15:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260302_03"
down_revision = "20260302_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "monologue_states" in existing_tables:
        columns = {col["name"] for col in inspector.get_columns("monologue_states")}
        if "user_state" not in columns:
            op.add_column(
                "monologue_states",
                sa.Column(
                    "user_state",
                    postgresql.JSONB(astext_type=sa.Text()),
                    nullable=False,
                    server_default="[]",
                ),
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "monologue_states" in existing_tables:
        columns = {col["name"] for col in inspector.get_columns("monologue_states")}
        if "user_state" in columns:
            op.drop_column("monologue_states", "user_state")
