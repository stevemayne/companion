"""add multi-character support columns

- episodic_messages: add nullable name column
- monologue_states: add nullable character_name column,
  change PK to composite (chat_session_id, character_name)
- session_seed_contexts: add characters JSONB column

Revision ID: 20260305_01
Revises: 20260302_03
Create Date: 2026-03-05 16:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260305_01"
down_revision = "20260302_03"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    # 1. episodic_messages: add name column
    if "episodic_messages" in existing_tables:
        columns = {col["name"] for col in inspector.get_columns("episodic_messages")}
        if "name" not in columns:
            op.add_column(
                "episodic_messages",
                sa.Column("name", sa.Text(), nullable=True),
            )

    # 2. monologue_states: add character_name and rebuild PK
    if "monologue_states" in existing_tables:
        columns = {col["name"] for col in inspector.get_columns("monologue_states")}
        if "character_name" not in columns:
            op.add_column(
                "monologue_states",
                sa.Column("character_name", sa.Text(), nullable=True),
            )
            # Drop old single-column PK and create composite unique constraint.
            # ON CONFLICT needs a unique index, not just a PK, so we use a
            # unique index that treats NULLs as equal via COALESCE.
            op.drop_constraint(
                "monologue_states_pkey", "monologue_states", type_="primary"
            )
            op.execute(
                "CREATE UNIQUE INDEX uq_monologue_session_character "
                "ON monologue_states (chat_session_id, COALESCE(character_name, ''))"
            )

    # 3. session_seed_contexts: add characters column
    if "session_seed_contexts" in existing_tables:
        columns = {
            col["name"] for col in inspector.get_columns("session_seed_contexts")
        }
        if "characters" not in columns:
            op.add_column(
                "session_seed_contexts",
                sa.Column(
                    "characters",
                    postgresql.JSONB(astext_type=sa.Text()),
                    nullable=False,
                    server_default="[]",
                ),
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "session_seed_contexts" in existing_tables:
        columns = {
            col["name"] for col in inspector.get_columns("session_seed_contexts")
        }
        if "characters" in columns:
            op.drop_column("session_seed_contexts", "characters")

    if "monologue_states" in existing_tables:
        columns = {col["name"] for col in inspector.get_columns("monologue_states")}
        if "character_name" in columns:
            op.execute("DROP INDEX IF EXISTS uq_monologue_session_character")
            op.drop_column("monologue_states", "character_name")
            # Restore original PK
            op.create_primary_key(
                "monologue_states_pkey",
                "monologue_states",
                ["chat_session_id"],
            )

    if "episodic_messages" in existing_tables:
        columns = {col["name"] for col in inspector.get_columns("episodic_messages")}
        if "name" in columns:
            op.drop_column("episodic_messages", "name")
