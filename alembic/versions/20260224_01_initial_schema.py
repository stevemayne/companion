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
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "episodic_messages" not in existing_tables:
        op.create_table(
            "episodic_messages",
            sa.Column("chat_session_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("message_id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("role", sa.Text(), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        )

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("episodic_messages")}
    if "idx_episodic_session_created" not in existing_indexes:
        op.create_index(
            "idx_episodic_session_created",
            "episodic_messages",
            ["chat_session_id", "created_at"],
            unique=False,
        )

    if "session_seed_contexts" not in existing_tables:
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
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "session_seed_contexts" in existing_tables:
        op.drop_table("session_seed_contexts")

    if "episodic_messages" in existing_tables:
        existing_indexes = {idx["name"] for idx in inspector.get_indexes("episodic_messages")}
        if "idx_episodic_session_created" in existing_indexes:
            op.drop_index("idx_episodic_session_created", table_name="episodic_messages")
        op.drop_table("episodic_messages")
