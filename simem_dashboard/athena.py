from __future__ import annotations

import time
from typing import Any

import boto3
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError


class AthenaQueryError(RuntimeError):
    """Error al consultar Athena."""


def _extract_value(cell: dict[str, Any]) -> str | None:
    value = cell.get("VarCharValue")
    return None if value in ("", None) else value


def run_query(
    sql: str,
    *,
    region_name: str,
    workgroup: str,
    output_location: str,
    poll_interval_seconds: float = 1.0,
) -> pd.DataFrame:
    try:
        client = boto3.client("athena", region_name=region_name)
        response = client.start_query_execution(
            QueryString=sql,
            WorkGroup=workgroup,
            ResultConfiguration={"OutputLocation": output_location},
        )
        execution_id = response["QueryExecutionId"]

        while True:
            execution = client.get_query_execution(QueryExecutionId=execution_id)
            state = execution["QueryExecution"]["Status"]["State"]

            if state == "SUCCEEDED":
                break
            if state in {"FAILED", "CANCELLED"}:
                reason = execution["QueryExecution"]["Status"].get("StateChangeReason", "Sin detalle")
                raise AthenaQueryError(f"Consulta fallida en Athena: {reason}")

            time.sleep(poll_interval_seconds)

        paginator = client.get_paginator("get_query_results")
        pages = paginator.paginate(QueryExecutionId=execution_id)

        columns: list[str] = []
        rows: list[list[str | None]] = []
        skip_header = True

        for page in pages:
            result_set = page["ResultSet"]
            if not columns:
                columns = [item["Name"] for item in result_set["ResultSetMetadata"]["ColumnInfo"]]

            for row in result_set["Rows"]:
                values = [_extract_value(cell) for cell in row.get("Data", [])]
                if skip_header:
                    skip_header = False
                    continue
                if len(values) < len(columns):
                    values.extend([None] * (len(columns) - len(values)))
                rows.append(values[: len(columns)])

        return pd.DataFrame(rows, columns=columns)
    except NoCredentialsError as error:
        raise AthenaQueryError(
            "No encontre credenciales AWS en la sesion para consultar Athena."
        ) from error
    except (BotoCoreError, ClientError) as error:
        raise AthenaQueryError(f"Error de conexion con Athena: {error}") from error
