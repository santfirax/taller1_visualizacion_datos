from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

from simem_dashboard.athena import run_query


SILVER_TABLES = (
    "aporte_hidricos",
    "demanda_comercial",
    "demanda_real",
    "generacion_real",
    "niveles_embalse",
    "precio_ponderado_bolsa",
    "unidades_generacion",
)

GOLD_TABLES = (
    "aporte_hidricos_agrupado",
    "co2_tasa_planta_mes",
    "demanda_real_hourly",
    "emisiones_sin_anual",
    "emisiones_sin_diario",
    "emisiones_sin_mensual",
    "generacion",
    "plantas_catalogo",
)


SEMANTIC_COLUMNS = {
    "date": (
        "fecha",
        "fechainicial",
        "fecha_inicial",
        "fechahora",
        "fecha_hora",
        "event_ts",
        "datetime",
        "period_start",
    ),
    "unit_id": (
        "codigounidadgeneracion",
        "codigo_unidad_generacion",
        "unidad",
    ),
    "generation_type": (
        "tipogeneracion",
        "tipo_generacion",
        "tecnologia",
    ),
    "resource_state": (
        "estadorecurso",
        "estado_recurso",
        "estado",
    ),
    "market_type": (
        "tipomercado",
        "tipo_mercado",
    ),
    "plant_code": (
        "codigoplanta",
        "codigo_planta",
        "planta",
    ),
    "generation_energy": (
        "valor",
        "energia",
        "energia_generada",
        "generacionreal",
        "generacion_real",
    ),
    "real_demand": (
        "demandaenergia",
        "demanda_real",
        "demanda",
        "energia",
        "valor",
    ),
    "commercial_demand": (
        "demanda_comercial",
        "demandaenergia",
        "demanda",
        "energia",
        "valor",
    ),
    "market_price": (
        "precioponderadobolsa",
        "precio_ponderado_bolsa",
        "precio",
        "valor",
    ),
    "emissions_value": (
        "emisiones_tco2",
        "emisiones_co2",
    ),
    "hydro_region": (
        "regionhidrologica",
        "region_hidrologica",
        "region",
    ),
    "hydro_actual": (
        "aportehidricoenergiakwhdia",
        "aportehidricoenergia",
        "aporteshidricosenergia",
        "aporte_energia",
        "aporte",
        "valor",
    ),
    "hydro_average": (
        "mediahistorica",
        "mediahistoricaenergia",
        "media_historica",
        "promediohistorico",
        "promedio_historico",
        "promedioacumuladoenergia",
    ),
    "reservoir_name": (
        "nombreembalse",
        "embalse",
        "nombre_embalse",
    ),
    "reservoir_level": (
        "porcentajevolumenutildiario",
        "porcentaje_volumen_util_diario",
        "porcentajevolumenutil",
        "nivelporcentual",
        "nivel_porcentual",
        "nivel",
        "valor",
    ),
}


NUMERIC_TYPES = {
    "tinyint",
    "smallint",
    "integer",
    "int",
    "bigint",
    "real",
    "double",
    "float",
    "decimal",
}


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    data_type: str


@dataclass(frozen=True)
class TableSchema:
    table_name: str
    columns: tuple[ColumnSchema, ...]

    def column_names(self) -> list[str]:
        return [column.name for column in self.columns]

    def get_column(self, name: str) -> ColumnSchema | None:
        for column in self.columns:
            if column.name == name:
                return column
        return None


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def build_metric_expression(column: ColumnSchema) -> str:
    identifier = quote_identifier(column.name)
    if any(column.data_type.startswith(item) for item in NUMERIC_TYPES):
        return f"TRY_CAST({identifier} AS double)"
    return f"TRY_CAST({identifier} AS double)"


def build_time_bucket(column: ColumnSchema) -> str:
    identifier = quote_identifier(column.name)
    lowered = column.data_type.lower()

    if lowered.startswith("timestamp"):
        return f"date({identifier})"
    if lowered.startswith("date"):
        return identifier
    return f"date(TRY_CAST({identifier} AS timestamp))"


def build_hour_bucket(column: ColumnSchema) -> str:
    identifier = quote_identifier(column.name)
    lowered = column.data_type.lower()

    if lowered.startswith("timestamp"):
        return f"hour({identifier})"
    if lowered.startswith("date"):
        return "0"
    return f"hour(TRY_CAST({identifier} AS timestamp))"


def build_timestamp_expression(column: ColumnSchema) -> str:
    identifier = quote_identifier(column.name)
    lowered = column.data_type.lower()

    if lowered.startswith("timestamp"):
        return identifier
    if lowered.startswith("date"):
        return f"CAST({identifier} AS timestamp)"
    return f"TRY_CAST({identifier} AS timestamp)"


def build_date_clause(expression: str, start_date: date | None, end_date: date | None) -> str:
    clauses = []
    if start_date:
        clauses.append(f"{expression} >= DATE '{start_date.isoformat()}'")
    if end_date:
        clauses.append(f"{expression} <= DATE '{end_date.isoformat()}'")
    return " AND ".join(clauses)


def build_in_clause(column_name: str, values: Iterable[str], case_insensitive: bool = False) -> str:
    items = [item for item in values if item]
    if not items:
        return ""

    quoted_values = ", ".join("'" + item.replace("'", "''") + "'" for item in items)
    identifier = quote_identifier(column_name)
    if case_insensitive:
        lowered_values = ", ".join("'" + item.lower().replace("'", "''") + "'" for item in items)
        return f"lower(CAST({identifier} AS varchar)) IN ({lowered_values})"
    return f"CAST({identifier} AS varchar) IN ({quoted_values})"


def choose_column(schema: TableSchema, semantic_key: str) -> ColumnSchema | None:
    candidates = SEMANTIC_COLUMNS.get(semantic_key, ())
    for candidate in candidates:
        column = schema.get_column(candidate)
        if column:
            return column
    return None


def discover_schemas(
    *,
    region_name: str,
    workgroup: str,
    output_location: str,
    database: str,
    table_names: Iterable[str],
) -> dict[str, TableSchema]:
    table_filter = ", ".join("'" + item.replace("'", "''") + "'" for item in table_names)
    sql = f"""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = '{database}'
          AND table_name IN ({table_filter})
        ORDER BY table_name, ordinal_position
    """
    dataframe = run_query(
        sql,
        region_name=region_name,
        workgroup=workgroup,
        output_location=output_location,
    )

    grouped: dict[str, list[ColumnSchema]] = {}
    for row in dataframe.to_dict(orient="records"):
        grouped.setdefault(row["table_name"], []).append(
            ColumnSchema(name=row["column_name"], data_type=row["data_type"])
        )

    return {
        table_name: TableSchema(table_name=table_name, columns=tuple(columns))
        for table_name, columns in grouped.items()
    }
