"""add_stock_sentiments

Revision ID: 4d66f10732b3
Revises: e539934b0909
Create Date: 2026-06-11

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision: str = "4d66f10732b3"
down_revision: str | None = "e539934b0909"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "stock_sentiments",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("date", sa.String(10), nullable=False),
        sa.Column("avg_sentiment", sa.Float, nullable=False),
        sa.Column("time_weighted_avg", sa.Float, nullable=True),
        sa.Column("raw_avg", sa.Float, nullable=True),
        sa.Column("impact_score_avg", sa.Float, nullable=True),
        sa.Column("news_count", sa.Integer, nullable=True),
        sa.Column("positive_pct", sa.Float, nullable=True),
        sa.Column("negative_pct", sa.Float, nullable=True),
        sa.Column("neutral_pct", sa.Float, nullable=True),
        sa.Column("signal", sa.String(20), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_stock_sentiment_ticker_date", "stock_sentiments", ["ticker", "date"])


def downgrade() -> None:
    op.drop_index("idx_stock_sentiment_ticker_date", "stock_sentiments")
    op.drop_table("stock_sentiments")