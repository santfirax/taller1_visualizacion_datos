from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DashboardSettings:
    region_name: str
    workgroup: str
    output_location: str
    database: str
    gold_database: str
    poll_interval_seconds: float


def load_settings() -> DashboardSettings:
    return DashboardSettings(
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        workgroup=os.getenv("ATHENA_WORKGROUP", "primary"),
        output_location=os.getenv(
            "ATHENA_OUTPUT",
            "s3://eafit-proyecto-integrador-simem/athena-results/",
        ),
        database=os.getenv("ATHENA_DATABASE", "simem_silver"),
        gold_database=os.getenv("ATHENA_GOLD_DATABASE", "simem_gold"),
        poll_interval_seconds=float(os.getenv("ATHENA_POLL_INTERVAL", "1.0")),
    )
