"""add monologue_states table and user_description column

Revision ID: 20260302_01
Revises: 20260224_01
Create Date: 2026-03-02 00:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260302_01"
down_revision = "20260224_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    # Add user_description to session_seed_contexts if missing
    if "session_seed_contexts" in existing_tables:
        columns = {col["name"] for col in inspector.get_columns("session_seed_contexts")}
        if "user_description" not in columns:
            op.add_column(
                "session_seed_contexts",
                sa.Column("user_description", sa.Text(), nullable=True),
            )

    # Create monologue_states table
    if "monologue_states" not in existing_tables:
        op.create_table(
            "monologue_states",
            sa.Column("chat_session_id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("internal_monologue", sa.Text(), nullable=False, server_default=""),
            sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "monologue_states" in existing_tables:
        op.drop_table("monologue_states")

    if "session_seed_contexts" in existing_tables:
        columns = {col["name"] for col in inspector.get_columns("session_seed_contexts")}
        if "user_description" in columns:
            op.drop_column("session_seed_contexts", "user_description")
