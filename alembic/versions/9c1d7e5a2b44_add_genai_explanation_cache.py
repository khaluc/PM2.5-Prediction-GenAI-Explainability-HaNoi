"""add persistent GenAI explanation cache

Revision ID: 9c1d7e5a2b44
Revises: 5678504a8fc8
Create Date: 2026-07-23 17:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "9c1d7e5a2b44"
down_revision: Union[str, None] = "5678504a8fc8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "genai_explanations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("station_id", sa.String(length=64), nullable=False),
        sa.Column("forecast_issued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("horizon_hours", sa.Integer(), nullable=False),
        sa.Column("cache_version", sa.String(length=80), nullable=False),
        sa.Column("generation_mode", sa.String(length=40), nullable=False),
        sa.Column("provider_model", sa.String(length=120), nullable=True),
        sa.Column("fallback_reason", sa.String(length=160), nullable=True),
        sa.Column(
            "result",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()),
                "postgresql",
            ),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "horizon_hours > 0",
            name=op.f("ck_genai_explanations_genai_horizon_positive"),
        ),
        sa.ForeignKeyConstraint(
            ["station_id"],
            ["stations.station_id"],
            name=op.f("fk_genai_explanations_station_id_stations"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_genai_explanations")),
        sa.UniqueConstraint(
            "station_id",
            "forecast_issued_at",
            "horizon_hours",
            "cache_version",
            name="genai_station_issue_horizon_version",
        ),
    )
    op.create_index(
        "ix_genai_station_issued_desc",
        "genai_explanations",
        ["station_id", sa.literal_column("forecast_issued_at DESC")],
        unique=False,
    )
    op.create_index(
        "ix_genai_expires_at",
        "genai_explanations",
        ["expires_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_genai_expires_at", table_name="genai_explanations")
    op.drop_index("ix_genai_station_issued_desc", table_name="genai_explanations")
    op.drop_table("genai_explanations")
