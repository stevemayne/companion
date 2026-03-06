"""add companion_id to support multiple companions per session

- episodic_messages: add speaker_id and speaker_name columns
- monologue_states: add companion_id, change PK to composite
- session_seed_contexts: add companion_id, change PK to composite

Revision ID: 20260306_01
Revises: 20260302_03
Create Date: 2026-03-06
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "20260306_01"
down_revision = "20260302_03"
branch_labels = None
depends_on = None

# Default companion_id for existing rows (deterministic so it's idempotent)
_DEFAULT_COMPANION_ID = "00000000-0000-0000-0000-000000000001"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # --- episodic_messages ---
    columns = {col["name"] for col in inspector.get_columns("episodic_messages")}
    if "speaker_id" not in columns:
        op.add_column(
            "episodic_messages",
            sa.Column("speaker_id", postgresql.UUID(as_uuid=True), nullable=True),
        )
    if "speaker_name" not in columns:
        op.add_column(
            "episodic_messages",
            sa.Column("speaker_name", sa.Text(), nullable=True),
        )

    # --- monologue_states ---
    columns = {col["name"] for col in inspector.get_columns("monologue_states")}
    if "companion_id" not in columns:
        op.add_column(
            "monologue_states",
            sa.Column(
                "companion_id",
                postgresql.UUID(as_uuid=True),
                nullable=True,
            ),
        )
        # Backfill existing rows with default companion_id
        op.execute(
            f"UPDATE monologue_states SET companion_id = '{_DEFAULT_COMPANION_ID}' "
            "WHERE companion_id IS NULL"
        )
        op.alter_column("monologue_states", "companion_id", nullable=False)

        # Rebuild PK as composite (chat_session_id, companion_id)
        op.drop_constraint("monologue_states_pkey", "monologue_states", type_="primary")
        op.create_primary_key(
            "monologue_states_pkey",
            "monologue_states",
            ["chat_session_id", "companion_id"],
        )

    # --- session_seed_contexts ---
    columns = {col["name"] for col in inspector.get_columns("session_seed_contexts")}
    if "companion_id" not in columns:
        op.add_column(
            "session_seed_contexts",
            sa.Column(
                "companion_id",
                postgresql.UUID(as_uuid=True),
                nullable=True,
            ),
        )
        # Backfill existing rows
        op.execute(
            f"UPDATE session_seed_contexts SET companion_id = '{_DEFAULT_COMPANION_ID}' "
            "WHERE companion_id IS NULL"
        )
        op.alter_column("session_seed_contexts", "companion_id", nullable=False)

        # Change PK from (chat_session_id) to (chat_session_id, companion_id)
        op.drop_constraint(
            "session_seed_contexts_pkey", "session_seed_contexts", type_="primary"
        )
        op.create_primary_key(
            "session_seed_contexts_pkey",
            "session_seed_contexts",
            ["chat_session_id", "companion_id"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # --- session_seed_contexts ---
    columns = {col["name"] for col in inspector.get_columns("session_seed_contexts")}
    if "companion_id" in columns:
        # Remove duplicates keeping lowest companion_id per session
        op.execute(
            "DELETE FROM session_seed_contexts a USING session_seed_contexts b "
            "WHERE a.chat_session_id = b.chat_session_id "
            "AND a.companion_id > b.companion_id"
        )
        op.drop_constraint(
            "session_seed_contexts_pkey", "session_seed_contexts", type_="primary"
        )
        op.create_primary_key(
            "session_seed_contexts_pkey",
            "session_seed_contexts",
            ["chat_session_id"],
        )
        op.drop_column("session_seed_contexts", "companion_id")

    # --- monologue_states ---
    columns = {col["name"] for col in inspector.get_columns("monologue_states")}
    if "companion_id" in columns:
        # Remove duplicates keeping lowest companion_id per session
        op.execute(
            "DELETE FROM monologue_states a USING monologue_states b "
            "WHERE a.chat_session_id = b.chat_session_id "
            "AND a.companion_id > b.companion_id"
        )
        op.drop_constraint("monologue_states_pkey", "monologue_states", type_="primary")
        op.create_primary_key(
            "monologue_states_pkey",
            "monologue_states",
            ["chat_session_id"],
        )
        op.drop_column("monologue_states", "companion_id")

    # --- episodic_messages ---
    columns = {col["name"] for col in inspector.get_columns("episodic_messages")}
    if "speaker_name" in columns:
        op.drop_column("episodic_messages", "speaker_name")
    if "speaker_id" in columns:
        op.drop_column("episodic_messages", "speaker_id")
