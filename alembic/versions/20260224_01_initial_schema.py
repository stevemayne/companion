"""initial schema

Revision ID: 20260224_01
Revises: 
Create Date: 2026-02-24 22:55:00
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260224_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "episodic_messages",
        sa.Column("chat_session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
    )
    op.create_index(
        "idx_episodic_session_created",
        "episodic_messages",
        ["chat_session_id", "created_at"],
        unique=False,
    )

    op.create_table(
        "session_seed_contexts",
        sa.Column("chat_session_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("companion_name", sa.Text(), nullable=False),
        sa.Column("backstory", sa.Text(), nullable=False),
        sa.Column("character_traits", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("goals", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("relationship_setup", sa.Text(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("session_seed_contexts")
    op.drop_index("idx_episodic_session_created", table_name="episodic_messages")
    op.drop_table("episodic_messages")
