"""add affect column to monologue_states

Revision ID: 20260302_02
Revises: 20260302_01
Create Date: 2026-03-02 12:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260302_02"
down_revision = "20260302_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "monologue_states" in existing_tables:
        columns = {col["name"] for col in inspector.get_columns("monologue_states")}
        if "affect" not in columns:
            op.add_column(
                "monologue_states",
                sa.Column(
                    "affect",
                    postgresql.JSONB(astext_type=sa.Text()),
                    nullable=False,
                    server_default="{}",
                ),
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "monologue_states" in existing_tables:
        columns = {col["name"] for col in inspector.get_columns("monologue_states")}
        if "affect" in columns:
            op.drop_column("monologue_states", "affect")
