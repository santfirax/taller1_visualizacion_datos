from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from simem_dashboard.athena import AthenaQueryError, run_query
from simem_dashboard.schema import (
    GOLD_TABLES,
    SILVER_TABLES,
    TableSchema,
    build_date_clause,
    build_hour_bucket,
    build_in_clause,
    build_metric_expression,
    build_time_bucket,
    build_timestamp_expression,
    choose_column,
    discover_schemas,
)
from simem_dashboard.settings import load_settings


st.set_page_config(
    page_title="SIMEM Silver Dashboards",
    page_icon=":bar_chart:",
    layout="wide",
)


DAY_LABELS = {
    1: "Lun",
    2: "Mar",
    3: "Mie",
    4: "Jue",
    5: "Vie",
    6: "Sab",
    7: "Dom",
}

GENERATION_BUCKET_ORDER = [
    "Termica",
    "Hidraulica",
    "Solar",
    "Eolica",
    "Cogeneracion",
    "Biomasa",
    "Otros",
    "Sin clasificar",
]

THERMAL_KEYWORDS = ("term", "gas", "carbon", "carb", "combust", "diesel", "acpm", "fuel", "jet")
HYDRO_KEYWORDS = ("hidr", "hydro", "agua")
SOLAR_KEYWORDS = ("solar", "fotovolta")
WIND_KEYWORDS = ("eolic", "viento", "wind")
COGENERATION_KEYWORDS = ("cogen",)
BIOMASS_KEYWORDS = ("biom", "bagazo")


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _format_date_bounds(bounds: tuple[date | None, date | None]) -> str:
    start_date, end_date = bounds
    if not start_date or not end_date:
        return "Sin fechas detectadas"
    return f"{start_date.isoformat()} a {end_date.isoformat()}"


def _format_range_label(start_date: date | None, end_date: date | None) -> str:
    if not start_date or not end_date:
        return "sin rango"
    return f"{start_date.isoformat()} a {end_date.isoformat()}"


def _format_number(value: float | None, decimals: int = 1, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "N/D"
    return f"{value:,.{decimals}f}{suffix}"


def _percentage_delta(current: float | None, previous: float | None) -> float | None:
    if current is None or previous in (None, 0) or pd.isna(current) or pd.isna(previous):
        return None
    return ((current - previous) / previous) * 100


def _delta_label(current: float | None, previous: float | None) -> str | None:
    delta = _percentage_delta(current, previous)
    if delta is None:
        return None
    return f"{delta:+.1f}% vs periodo anterior"


def _calculate_previous_period(
    start_date: date | None, end_date: date | None
) -> tuple[date | None, date | None]:
    if not start_date or not end_date:
        return None, None
    days = (end_date - start_date).days + 1
    previous_end = start_date - timedelta(days=1)
    previous_start = previous_end - timedelta(days=days - 1)
    return previous_start, previous_end


def _coerce_date_range(
    raw_value: object, default_start: date | None, default_end: date | None
) -> tuple[date | None, date | None]:
    if (
        isinstance(raw_value, (tuple, list))
        and len(raw_value) == 2
        and isinstance(raw_value[0], date)
        and isinstance(raw_value[1], date)
    ):
        return raw_value[0], raw_value[1]
    return default_start, default_end


def _clamp_date(value: date, minimum: date | None, maximum: date | None) -> date:
    if minimum and value < minimum:
        return minimum
    if maximum and value > maximum:
        return maximum
    return value


def _classify_generation_bucket(raw_value: object) -> str:
    if raw_value is None or pd.isna(raw_value):
        return "Sin clasificar"

    normalized = str(raw_value).strip().lower()
    if not normalized:
        return "Sin clasificar"
    if any(keyword in normalized for keyword in THERMAL_KEYWORDS):
        return "Termica"
    if any(keyword in normalized for keyword in HYDRO_KEYWORDS):
        return "Hidraulica"
    if any(keyword in normalized for keyword in SOLAR_KEYWORDS):
        return "Solar"
    if any(keyword in normalized for keyword in WIND_KEYWORDS):
        return "Eolica"
    if any(keyword in normalized for keyword in COGENERATION_KEYWORDS):
        return "Cogeneracion"
    if any(keyword in normalized for keyword in BIOMASS_KEYWORDS):
        return "Biomasa"
    return "Otros"


def _add_period_bucket(dataframe: pd.DataFrame, frequency: str) -> pd.DataFrame:
    result = dataframe.copy()
    if result.empty or "periodo" not in result.columns:
        return result

    result["periodo"] = pd.to_datetime(result["periodo"], errors="coerce")
    if frequency == "MS":
        result["periodo_bucket"] = result["periodo"].dt.to_period("M").dt.to_timestamp()
    else:
        result["periodo_bucket"] = result["periodo"].dt.to_period("W").dt.start_time
    return result.dropna(subset=["periodo_bucket"])


def _format_period_label(period_value: pd.Timestamp | None, granularity: str) -> str:
    if period_value is None or pd.isna(period_value):
        return "N/D"
    if granularity == "Mensual":
        return pd.Timestamp(period_value).strftime("%Y-%m")
    return pd.Timestamp(period_value).strftime("%Y-%m-%d")


def _resolve_highlight_window(
    trigger_period: pd.Timestamp | None,
    fallback_start: date | None,
    event_end: date | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if not event_end:
        return None, None

    start_ts = pd.Timestamp(trigger_period) if trigger_period is not None and not pd.isna(trigger_period) else None
    if start_ts is None and fallback_start is not None:
        start_ts = pd.Timestamp(fallback_start)

    end_ts = pd.Timestamp(event_end)
    if start_ts is not None and end_ts < start_ts:
        end_ts = start_ts
    return start_ts, end_ts


def _add_time_window_highlight(
    figure: go.Figure,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
    label: str = "Fenomeno El Nino",
) -> None:
    if start_ts is None or end_ts is None or pd.isna(start_ts) or pd.isna(end_ts):
        return

    start_value = pd.Timestamp(start_ts)
    end_value = pd.Timestamp(end_ts)
    if end_value < start_value:
        end_value = start_value

    figure.add_shape(
        type="rect",
        x0=start_value,
        x1=end_value,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        fillcolor="rgba(214, 192, 132, 0.16)",
        line=dict(width=0),
        layer="below",
    )

    midpoint = start_value + ((end_value - start_value) / 2 if end_value > start_value else pd.Timedelta(0))
    figure.add_annotation(
        x=midpoint,
        y=1.04,
        xref="x",
        yref="paper",
        text=label,
        showarrow=False,
        font=dict(color="#d6c6a5", size=11),
        bgcolor="rgba(120, 113, 108, 0.35)",
        bordercolor="rgba(214, 198, 165, 0.35)",
        borderpad=4,
    )


def _normalize_series(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    if "periodo" in result.columns:
        result["periodo"] = pd.to_datetime(result["periodo"], errors="coerce")
    if "valor" in result.columns:
        result["valor"] = pd.to_numeric(result["valor"], errors="coerce")
    return result


def _extract_series_metric(dataframe: pd.DataFrame, series_name: str, reducer: str) -> float | None:
    if dataframe.empty:
        return None
    subset = dataframe[dataframe["serie"] == series_name]
    if subset.empty:
        return None
    values = pd.to_numeric(subset["valor"], errors="coerce").dropna()
    if values.empty:
        return None
    if reducer == "sum":
        return float(values.sum())
    if reducer == "mean":
        return float(values.mean())
    if reducer == "max":
        return float(values.max())
    return None


@st.cache_data(ttl=900, show_spinner=False)
def load_schemas(
    region_name: str,
    workgroup: str,
    output_location: str,
    database: str,
    table_names: tuple[str, ...] = SILVER_TABLES,
) -> dict[str, TableSchema]:
    return discover_schemas(
        region_name=region_name,
        workgroup=workgroup,
        output_location=output_location,
        database=database,
        table_names=table_names,
    )


@st.cache_data(ttl=900, show_spinner=False)
def load_table_values(
    region_name: str,
    workgroup: str,
    output_location: str,
    database: str,
    table_name: str,
    column_name: str,
    limit: int = 200,
) -> list[str]:
    sql = f"""
        SELECT DISTINCT CAST("{column_name}" AS varchar) AS value
        FROM "{database}"."{table_name}"
        WHERE "{column_name}" IS NOT NULL
        ORDER BY 1
        LIMIT {limit}
    """
    dataframe = run_query(sql, region_name=region_name, workgroup=workgroup, output_location=output_location)
    return [item for item in dataframe.get("value", pd.Series(dtype=str)).dropna().astype(str).tolist() if item]


@st.cache_data(ttl=900, show_spinner=False)
def load_date_bounds(
    region_name: str,
    workgroup: str,
    output_location: str,
    database: str,
    table_name: str,
    date_expression: str,
) -> tuple[date | None, date | None]:
    sql = f"""
        SELECT
            MIN({date_expression}) AS min_date,
            MAX({date_expression}) AS max_date
        FROM "{database}"."{table_name}"
        WHERE {date_expression} IS NOT NULL
    """
    dataframe = run_query(sql, region_name=region_name, workgroup=workgroup, output_location=output_location)
    if dataframe.empty:
        return None, None

    start_value = pd.to_datetime(dataframe.iloc[0]["min_date"], errors="coerce")
    end_value = pd.to_datetime(dataframe.iloc[0]["max_date"], errors="coerce")
    if pd.isna(start_value) or pd.isna(end_value):
        return None, None
    return start_value.date(), end_value.date()


@st.cache_data(ttl=900, show_spinner=False)
def load_dataframe(sql: str, region_name: str, workgroup: str, output_location: str) -> pd.DataFrame:
    return run_query(sql, region_name=region_name, workgroup=workgroup, output_location=output_location)


def _build_where(clauses: Iterable[str]) -> str:
    cleaned = [clause for clause in clauses if clause]
    return f"WHERE {' AND '.join(cleaned)}" if cleaned else ""


def render_connection_status(settings) -> None:
    with st.expander("Configuracion de conexion", expanded=False):
        st.write(
            {
                "region": settings.region_name,
                "workgroup": settings.workgroup,
                "silver_database": settings.database,
                "gold_database": settings.gold_database,
                "output_location": settings.output_location,
            }
        )


def render_schema_warning(table_name: str, schema: TableSchema, missing: list[str]) -> None:
    st.warning(
        f"La tabla `{table_name}` no tiene todas las columnas esperadas para este visual. "
        f"Faltan: {', '.join(missing)}. Columnas detectadas: {', '.join(schema.column_names())}"
    )


def build_units_distribution_query(
    database: str,
    schema: TableSchema,
    generation_types: list[str],
    resource_states: list[str],
) -> str | None:
    unit_id = choose_column(schema, "unit_id")
    generation_type = choose_column(schema, "generation_type")
    resource_state = choose_column(schema, "resource_state")

    missing = []
    if not unit_id:
        missing.append("unit_id")
    if not generation_type:
        missing.append("generation_type")
    if not resource_state:
        missing.append("resource_state")
    if missing:
        render_schema_warning("unidades_generacion", schema, missing)
        return None

    clauses = []
    if resource_states:
        clauses.append(build_in_clause(resource_state.name, resource_states, case_insensitive=True))
    if generation_types:
        clauses.append(build_in_clause(generation_type.name, generation_types))

    where_sql = _build_where(clauses)
    return f"""
        SELECT
            "{generation_type.name}" AS tipo_generacion,
            COUNT(DISTINCT "{unit_id.name}") AS unidades
        FROM "{database}"."{schema.table_name}"
        {where_sql}
        GROUP BY 1
        ORDER BY 2 DESC, 1
    """


def build_units_top_plants_query(
    database: str,
    schema: TableSchema,
    generation_types: list[str],
    resource_states: list[str],
    limit: int = 15,
) -> str | None:
    unit_id = choose_column(schema, "unit_id")
    plant_code = choose_column(schema, "plant_code")
    generation_type = choose_column(schema, "generation_type")
    resource_state = choose_column(schema, "resource_state")
    if not unit_id or not plant_code:
        return None

    clauses = []
    if resource_state and resource_states:
        clauses.append(build_in_clause(resource_state.name, resource_states, case_insensitive=True))
    if generation_type and generation_types:
        clauses.append(build_in_clause(generation_type.name, generation_types))

    where_sql = _build_where(clauses)
    return f"""
        SELECT
            CAST("{plant_code.name}" AS varchar) AS planta,
            COUNT(DISTINCT "{unit_id.name}") AS unidades
        FROM "{database}"."{schema.table_name}"
        {where_sql}
        GROUP BY 1
        ORDER BY 2 DESC, 1
        LIMIT {limit}
    """


def build_daily_series_query(
    database: str,
    schema: TableSchema,
    metric_key: str,
    label: str,
    start_date: date | None,
    end_date: date | None,
    market_types: list[str] | None = None,
    aggregation: str = "SUM",
) -> str | None:
    date_column = choose_column(schema, "date")
    value_column = choose_column(schema, metric_key)
    if not date_column or not value_column:
        return None

    date_expression = build_time_bucket(date_column)
    metric_expression = build_metric_expression(value_column)
    clauses = [build_date_clause(date_expression, start_date, end_date)]

    market_type_column = choose_column(schema, "market_type")
    if market_type_column and market_types:
        clauses.append(build_in_clause(market_type_column.name, market_types))

    where_sql = _build_where(clauses)
    return f"""
        SELECT
            {date_expression} AS periodo,
            {aggregation}({metric_expression}) AS valor,
            {_sql_quote(label)} AS serie
        FROM "{database}"."{schema.table_name}"
        {where_sql}
        GROUP BY 1, 3
        ORDER BY 1
    """


def build_generation_mix_query(
    database: str,
    generation_schema: TableSchema,
    units_schema: TableSchema,
    start_date: date | None,
    end_date: date | None,
) -> str | None:
    generation_date = choose_column(generation_schema, "date")
    generation_plant = choose_column(generation_schema, "plant_code")
    generation_value = choose_column(generation_schema, "generation_energy")
    unit_plant = choose_column(units_schema, "plant_code")
    unit_generation_type = choose_column(units_schema, "generation_type")
    unit_date = choose_column(units_schema, "date")

    missing = []
    if not generation_date:
        missing.append("date")
    if not generation_plant:
        missing.append("plant_code")
    if not generation_value:
        missing.append("generation_energy")
    if not unit_plant:
        missing.append("plant_code")
    if not unit_generation_type:
        missing.append("generation_type")
    if missing:
        st.warning(
            "No pude construir el analisis de emisiones para `generacion_real` y `unidades_generacion`. "
            f"Faltan columnas semanticas: {', '.join(missing)}. "
            f"Generacion detectada: {', '.join(generation_schema.column_names())}. "
            f"Unidades detectadas: {', '.join(units_schema.column_names())}."
        )
        return None

    generation_date_expression = build_time_bucket(generation_date)
    generation_value_expression = build_metric_expression(generation_value)
    generation_clauses = [
        build_date_clause(generation_date_expression, start_date, end_date),
        f'"{generation_plant.name}" IS NOT NULL',
        f"{generation_value_expression} IS NOT NULL",
    ]
    generation_where = _build_where(generation_clauses)

    units_clauses = [
        f'"{unit_plant.name}" IS NOT NULL',
        f'"{unit_generation_type.name}" IS NOT NULL',
    ]
    if unit_date:
        units_clauses.append(
            build_date_clause(build_time_bucket(unit_date), start_date, end_date)
        )
    units_where = _build_where(units_clauses)

    return f"""
        WITH plant_type_counts AS (
            SELECT
                CAST("{unit_plant.name}" AS varchar) AS planta,
                CAST("{unit_generation_type.name}" AS varchar) AS tipo_generacion,
                COUNT(*) AS registros
            FROM "{database}"."{units_schema.table_name}"
            {units_where}
            GROUP BY 1, 2
        ),
        plant_type_map AS (
            SELECT planta, tipo_generacion
            FROM (
                SELECT
                    planta,
                    tipo_generacion,
                    registros,
                    ROW_NUMBER() OVER (
                        PARTITION BY planta
                        ORDER BY registros DESC, tipo_generacion
                    ) AS rn
                FROM plant_type_counts
            ) ranked
            WHERE rn = 1
        ),
        generation_base AS (
            SELECT
                {generation_date_expression} AS periodo,
                CAST("{generation_plant.name}" AS varchar) AS planta,
                SUM({generation_value_expression}) AS energia_kwh
            FROM "{database}"."{generation_schema.table_name}"
            {generation_where}
            GROUP BY 1, 2
        )
        SELECT
            generation_base.periodo,
            COALESCE(plant_type_map.tipo_generacion, 'Sin clasificar') AS tipo_generacion,
            SUM(generation_base.energia_kwh) AS energia_kwh
        FROM generation_base
        LEFT JOIN plant_type_map
            ON generation_base.planta = plant_type_map.planta
        GROUP BY 1, 2
        ORDER BY 1, 2
    """


def build_gold_emissions_query(
    database: str,
    schema: TableSchema,
    start_date: date | None,
    end_date: date | None,
    granularity: str,
) -> str | None:
    date_column = choose_column(schema, "date")
    emissions_column = choose_column(schema, "emissions_value")
    plant_column = schema.get_column("n_plantas")
    if not date_column or not emissions_column:
        return None

    timestamp_expression = build_timestamp_expression(date_column)
    date_expression = build_time_bucket(date_column)
    metric_expression = build_metric_expression(emissions_column)
    period_expression = (
        f"date_trunc('month', {timestamp_expression})"
        if granularity == "Mensual"
        else f"date_trunc('week', {timestamp_expression})"
    )
    plant_expression = (
        f"AVG(TRY_CAST(\"{plant_column.name}\" AS double)) AS n_plantas"
        if plant_column
        else "CAST(NULL AS double) AS n_plantas"
    )
    where_sql = _build_where(
        [
            f"{timestamp_expression} IS NOT NULL",
            build_date_clause(date_expression, start_date, end_date),
            f"{metric_expression} IS NOT NULL",
        ]
    )
    return f"""
        SELECT
            CAST({period_expression} AS timestamp) AS periodo,
            SUM({metric_expression}) AS emisiones_tco2,
            {plant_expression}
        FROM "{database}"."{schema.table_name}"
        {where_sql}
        GROUP BY 1
        ORDER BY 1
    """


def build_gold_generation_query(
    database: str,
    schema: TableSchema,
    start_date: date | None,
    end_date: date | None,
    granularity: str,
) -> str | None:
    date_column = schema.get_column("fechahora") or choose_column(schema, "date")
    value_column = choose_column(schema, "generation_energy")
    generation_type_column = choose_column(schema, "generation_type")
    if not date_column or not value_column or not generation_type_column:
        return None

    timestamp_expression = build_timestamp_expression(date_column)
    date_expression = build_time_bucket(date_column)
    metric_expression = build_metric_expression(value_column)
    period_expression = (
        f"date_trunc('month', {timestamp_expression})"
        if granularity == "Mensual"
        else f"date_trunc('week', {timestamp_expression})"
    )
    where_sql = _build_where(
        [
            f"{timestamp_expression} IS NOT NULL",
            build_date_clause(date_expression, start_date, end_date),
            f'"{generation_type_column.name}" IS NOT NULL',
            f"{metric_expression} IS NOT NULL",
        ]
    )
    return f"""
        SELECT
            CAST({period_expression} AS timestamp) AS periodo,
            CAST("{generation_type_column.name}" AS varchar) AS tipo_generacion,
            SUM({metric_expression}) AS energia_kwh
        FROM "{database}"."{schema.table_name}"
        {where_sql}
        GROUP BY 1, 2
        ORDER BY 1, 2
    """


def build_gold_hydrology_ratio_query(
    database: str,
    schema: TableSchema,
    start_date: date | None,
    end_date: date | None,
    granularity: str,
) -> str | None:
    timestamp_column = schema.get_column("event_ts")
    ratio_column = schema.get_column("aporte_hidrico_vs_media_ratio")
    if not timestamp_column or not ratio_column:
        return None

    timestamp_expression = build_timestamp_expression(timestamp_column)
    date_expression = build_time_bucket(timestamp_column)
    ratio_expression = build_metric_expression(ratio_column)
    period_expression = (
        f"date_trunc('month', {timestamp_expression})"
        if granularity == "Mensual"
        else f"date_trunc('week', {timestamp_expression})"
    )
    where_sql = _build_where(
        [
            f"{timestamp_expression} IS NOT NULL",
            build_date_clause(date_expression, start_date, end_date),
            f"{ratio_expression} IS NOT NULL",
        ]
    )
    return f"""
        SELECT
            CAST({period_expression} AS timestamp) AS periodo,
            AVG({ratio_expression}) AS ratio_hidrologico
        FROM "{database}"."{schema.table_name}"
        {where_sql}
        GROUP BY 1
        ORDER BY 1
    """


def build_gold_co2_drivers_query(
    database: str,
    co2_schema: TableSchema,
    plants_schema: TableSchema | None,
    start_date: date | None,
    end_date: date | None,
    limit: int = 12,
) -> str | None:
    plant_code_column = co2_schema.get_column("codigo_planta")
    year_column = co2_schema.get_column("anio")
    month_column = co2_schema.get_column("mes")
    emissions_column = choose_column(co2_schema, "emissions_value")
    generation_column = co2_schema.get_column("generacion_mwh")
    fuel_column = co2_schema.get_column("combustible_dom")
    generation_type_column = co2_schema.get_column("upme_tipo_generacion")
    rate_column = co2_schema.get_column("tasa_tco2_mwh")
    if not plant_code_column or not year_column or not month_column or not emissions_column:
        return None

    period_expression = (
        f"CAST(concat(CAST(\"{year_column.name}\" AS varchar), '-', lpad(CAST(\"{month_column.name}\" AS varchar), 2, '0'), '-01') AS date)"
    )
    emissions_expression = build_metric_expression(emissions_column)
    generation_expression = (
        build_metric_expression(generation_column)
        if generation_column
        else "CAST(NULL AS double)"
    )
    rate_expression = build_metric_expression(rate_column) if rate_column else "CAST(NULL AS double)"
    plants_join = ""
    plant_name_select = f'CAST(base.codigo_planta AS varchar) AS planta_nombre'
    if plants_schema and plants_schema.get_column("codigo_planta") and plants_schema.get_column("nombre_planta"):
        catalog_plant = plants_schema.get_column("codigo_planta")
        catalog_name = plants_schema.get_column("nombre_planta")
        plants_join = (
            f'LEFT JOIN "{database}"."{plants_schema.table_name}" catalog '
            f'ON base.codigo_planta = CAST(catalog."{catalog_plant.name}" AS varchar)'
        )
        plant_name_select = (
            f"COALESCE(CAST(catalog.\"{catalog_name.name}\" AS varchar), CAST(base.codigo_planta AS varchar)) "
            "AS planta_nombre"
        )

    where_sql = _build_where(
        [
            build_date_clause(period_expression, start_date, end_date),
            f"{emissions_expression} IS NOT NULL",
        ]
    )
    fuel_select = (
        f'COALESCE(CAST("{fuel_column.name}" AS varchar), \'Sin dato\') AS combustible_dom'
        if fuel_column
        else "'Sin dato' AS combustible_dom"
    )
    generation_type_select = (
        f'COALESCE(CAST("{generation_type_column.name}" AS varchar), \'Sin tipo\') AS tipo_generacion'
        if generation_type_column
        else "'Sin tipo' AS tipo_generacion"
    )
    return f"""
        WITH base AS (
            SELECT
                CAST("{plant_code_column.name}" AS varchar) AS codigo_planta,
                {fuel_select},
                {generation_type_select},
                SUM({emissions_expression}) AS emisiones_tco2,
                SUM({generation_expression}) AS generacion_mwh,
                AVG({rate_expression}) AS tasa_tco2_mwh
            FROM "{database}"."{co2_schema.table_name}"
            {where_sql}
            GROUP BY 1, 2, 3
        )
        SELECT
            {plant_name_select},
            base.codigo_planta,
            base.combustible_dom,
            base.tipo_generacion,
            base.emisiones_tco2,
            base.generacion_mwh,
            base.tasa_tco2_mwh
        FROM base
        {plants_join}
        ORDER BY base.emisiones_tco2 DESC
        LIMIT {limit}
    """


def build_duck_curve_query(
    database: str,
    schema: TableSchema,
    start_date: date | None,
    end_date: date | None,
    market_types: list[str] | None = None,
) -> str | None:
    date_column = choose_column(schema, "date")
    value_column = choose_column(schema, "real_demand")
    if not date_column or not value_column:
        return None

    date_expression = build_time_bucket(date_column)
    hour_expression = build_hour_bucket(date_column)
    metric_expression = build_metric_expression(value_column)
    clauses = [build_date_clause(date_expression, start_date, end_date)]

    market_type_column = choose_column(schema, "market_type")
    if market_type_column and market_types:
        clauses.append(build_in_clause(market_type_column.name, market_types))

    where_sql = _build_where(clauses)

    return f"""
        SELECT
            {hour_expression} AS hora,
            AVG({metric_expression}) AS demanda_promedio,
            MIN({metric_expression}) AS demanda_minima,
            MAX({metric_expression}) AS demanda_maxima
        FROM "{database}"."{schema.table_name}"
        {where_sql}
        GROUP BY 1
        ORDER BY 1
    """


def build_hourly_heatmap_query(
    database: str,
    schema: TableSchema,
    start_date: date | None,
    end_date: date | None,
    market_types: list[str] | None = None,
) -> str | None:
    date_column = choose_column(schema, "date")
    value_column = choose_column(schema, "real_demand")
    if not date_column or not value_column:
        return None

    timestamp_expression = build_timestamp_expression(date_column)
    date_expression = build_time_bucket(date_column)
    metric_expression = build_metric_expression(value_column)
    clauses = [
        f"{timestamp_expression} IS NOT NULL",
        build_date_clause(date_expression, start_date, end_date),
    ]

    market_type_column = choose_column(schema, "market_type")
    if market_type_column and market_types:
        clauses.append(build_in_clause(market_type_column.name, market_types))

    day_label_expression = (
        f"CASE day_of_week({timestamp_expression}) "
        "WHEN 1 THEN 'Lun' "
        "WHEN 2 THEN 'Mar' "
        "WHEN 3 THEN 'Mie' "
        "WHEN 4 THEN 'Jue' "
        "WHEN 5 THEN 'Vie' "
        "WHEN 6 THEN 'Sab' "
        "WHEN 7 THEN 'Dom' "
        "END"
    )
    where_sql = _build_where(clauses)

    return f"""
        SELECT
            day_of_week({timestamp_expression}) AS dia_num,
            {day_label_expression} AS dia_semana,
            hour({timestamp_expression}) AS hora,
            AVG({metric_expression}) AS demanda_promedio
        FROM "{database}"."{schema.table_name}"
        {where_sql}
        GROUP BY 1, 2, 3
        ORDER BY 1, 3
    """


def build_hydrology_query(
    database: str,
    schema: TableSchema,
    start_date: date | None,
    end_date: date | None,
    regions: list[str],
) -> str | None:
    date_column = choose_column(schema, "date")
    region_column = choose_column(schema, "hydro_region")
    actual_column = choose_column(schema, "hydro_actual")
    average_column = choose_column(schema, "hydro_average")

    missing = []
    if not date_column:
        missing.append("date")
    if not region_column:
        missing.append("hydro_region")
    if not actual_column:
        missing.append("hydro_actual")
    if missing:
        render_schema_warning("aporte_hidricos", schema, missing)
        return None

    date_expression = build_time_bucket(date_column)
    actual_expression = build_metric_expression(actual_column)
    average_expression = build_metric_expression(average_column) if average_column else None

    clauses = [build_date_clause(date_expression, start_date, end_date)]
    if regions:
        clauses.append(build_in_clause(region_column.name, regions))
    where_sql = _build_where(clauses)

    average_select = (
        f", AVG({average_expression}) AS media_historica"
        if average_expression
        else ", CAST(NULL AS double) AS media_historica"
    )
    return f"""
        SELECT
            {date_expression} AS periodo,
            CAST("{region_column.name}" AS varchar) AS region,
            SUM({actual_expression}) AS aporte_actual
            {average_select}
        FROM "{database}"."{schema.table_name}"
        {where_sql}
        GROUP BY 1, 2
        ORDER BY 1, 2
    """


def build_reservoir_query(
    database: str,
    schema: TableSchema,
    start_date: date | None,
    end_date: date | None,
) -> str | None:
    date_column = choose_column(schema, "date")
    reservoir_column = choose_column(schema, "reservoir_name")
    level_column = choose_column(schema, "reservoir_level")
    if not date_column or not reservoir_column or not level_column:
        return None

    date_expression = build_time_bucket(date_column)
    level_expression = build_metric_expression(level_column)
    where_sql = _build_where([build_date_clause(date_expression, start_date, end_date)])

    return f"""
        WITH series AS (
            SELECT
                {date_expression} AS periodo,
                CAST("{reservoir_column.name}" AS varchar) AS embalse,
                AVG({level_expression}) AS nivel
            FROM "{database}"."{schema.table_name}"
            {where_sql}
            GROUP BY 1, 2
        ),
        latest_point AS (
            SELECT MAX(periodo) AS latest_date FROM series
        )
        SELECT
            series.periodo,
            series.embalse,
            series.nivel,
            CASE WHEN series.periodo = latest_point.latest_date THEN TRUE ELSE FALSE END AS es_ultimo_corte
        FROM series
        CROSS JOIN latest_point
        ORDER BY series.periodo, series.embalse
    """


def build_demand_price_scatter(demand_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    if demand_df.empty or price_df.empty:
        return pd.DataFrame()

    demand_wide = (
        demand_df.pivot_table(index="periodo", columns="serie", values="valor", aggfunc="sum")
        .reset_index()
        .rename_axis(columns=None)
    )
    price_wide = (
        price_df.pivot_table(index="periodo", columns="serie", values="valor", aggfunc="mean")
        .reset_index()
        .rename_axis(columns=None)
    )
    merged = demand_wide.merge(price_wide, on="periodo", how="inner")
    if merged.empty:
        return merged

    if "Demanda real" in merged.columns:
        merged["demanda"] = merged["Demanda real"]
    elif "Demanda comercial" in merged.columns:
        merged["demanda"] = merged["Demanda comercial"]

    if "Precio bolsa" in merged.columns:
        merged["precio_bolsa"] = merged["Precio bolsa"]

    return merged.dropna(subset=["demanda", "precio_bolsa"], how="any")


def build_comparison_dataframe(
    current_demand_df: pd.DataFrame,
    previous_demand_df: pd.DataFrame,
    current_price_df: pd.DataFrame,
    previous_price_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "metrica": "Demanda real total",
            "periodo_actual": _extract_series_metric(current_demand_df, "Demanda real", "sum"),
            "periodo_anterior": _extract_series_metric(previous_demand_df, "Demanda real", "sum"),
        },
        {
            "metrica": "Demanda comercial total",
            "periodo_actual": _extract_series_metric(current_demand_df, "Demanda comercial", "sum"),
            "periodo_anterior": _extract_series_metric(previous_demand_df, "Demanda comercial", "sum"),
        },
        {
            "metrica": "Precio promedio bolsa",
            "periodo_actual": _extract_series_metric(current_price_df, "Precio bolsa", "mean"),
            "periodo_anterior": _extract_series_metric(previous_price_df, "Precio bolsa", "mean"),
        },
    ]
    dataframe = pd.DataFrame(rows)
    dataframe["variacion_pct"] = dataframe.apply(
        lambda row: _percentage_delta(row["periodo_actual"], row["periodo_anterior"]),
        axis=1,
    )
    return dataframe


def prepare_units_breakdown(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame()

    result = dataframe.copy()
    result["unidades"] = pd.to_numeric(result["unidades"], errors="coerce")
    result = result.dropna(subset=["unidades"]).sort_values("unidades", ascending=False)
    if result.empty:
        return result

    total_units = float(result["unidades"].sum())
    result["participacion_pct"] = (
        (result["unidades"] / total_units) * 100 if total_units else 0
    )
    result["participacion_label"] = result["participacion_pct"].map(lambda value: f"{value:.1f}%")
    result["unidades_label"] = result["unidades"].map(lambda value: f"{value:,.0f}")
    return result


def build_market_indexed_dataframe(demand_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    merged = build_demand_price_scatter(demand_df, price_df)
    if merged.empty:
        return pd.DataFrame()

    merged = merged.sort_values("periodo").copy()
    base_demand = merged["demanda"].iloc[0]
    base_price = merged["precio_bolsa"].iloc[0]
    if pd.isna(base_demand) or pd.isna(base_price) or base_demand == 0 or base_price == 0:
        return pd.DataFrame()

    merged["Demanda indexada"] = (merged["demanda"] / base_demand) * 100
    merged["Precio indexado"] = (merged["precio_bolsa"] / base_price) * 100
    return merged.melt(
        id_vars="periodo",
        value_vars=["Demanda indexada", "Precio indexado"],
        var_name="indicador",
        value_name="indice",
    )


def prepare_el_nino_analysis(
    emissions_df: pd.DataFrame,
    generation_df: pd.DataFrame,
    hydrology_df: pd.DataFrame,
    drivers_df: pd.DataFrame,
    baseline_end_date: date | None,
    granularity: str,
) -> dict[str, object]:
    empty_result: dict[str, object] = {
        "summary": pd.DataFrame(),
        "mix": pd.DataFrame(),
        "drivers": pd.DataFrame(),
        "baseline_share_pct": None,
        "baseline_emissions_tco2": None,
        "baseline_intensity_tco2_mwh": None,
        "trigger_period": None,
        "coverage_pct": None,
        "hydrology_available": False,
        "gold_source": True,
    }

    if emissions_df.empty or generation_df.empty or baseline_end_date is None:
        return empty_result

    frequency = "MS" if granularity == "Mensual" else "W"
    generation_base = generation_df.copy()
    generation_base["periodo"] = pd.to_datetime(generation_base["periodo"], errors="coerce")
    generation_base["energia_kwh"] = pd.to_numeric(generation_base["energia_kwh"], errors="coerce")
    generation_base = generation_base.dropna(subset=["periodo", "energia_kwh"])
    if generation_base.empty:
        return empty_result

    generation_base["bucket"] = generation_base["tipo_generacion"].map(_classify_generation_bucket)
    generation_base = _add_period_bucket(generation_base, frequency)
    if generation_base.empty:
        return empty_result

    mix_df = (
        generation_base.groupby(["periodo_bucket", "bucket"], as_index=False)["energia_kwh"].sum()
        .rename(columns={"periodo_bucket": "periodo"})
    )
    mix_df["bucket"] = pd.Categorical(
        mix_df["bucket"], categories=GENERATION_BUCKET_ORDER, ordered=True
    )
    mix_df = mix_df.sort_values(["periodo", "bucket"])

    generation_summary = (
        mix_df.groupby("periodo", as_index=False)["energia_kwh"].sum().rename(columns={"energia_kwh": "energia_total_kwh"})
    )
    thermal_df = (
        mix_df[mix_df["bucket"] == "Termica"]
        .groupby("periodo", as_index=False)["energia_kwh"]
        .sum()
        .rename(columns={"energia_kwh": "energia_termica_kwh"})
    )
    hydro_df = (
        mix_df[mix_df["bucket"] == "Hidraulica"]
        .groupby("periodo", as_index=False)["energia_kwh"]
        .sum()
        .rename(columns={"energia_kwh": "energia_hidraulica_kwh"})
    )
    unclassified_df = (
        mix_df[mix_df["bucket"] == "Sin clasificar"]
        .groupby("periodo", as_index=False)["energia_kwh"]
        .sum()
        .rename(columns={"energia_kwh": "energia_sin_clasificar_kwh"})
    )

    generation_summary = generation_summary.merge(thermal_df, on="periodo", how="left").merge(
        hydro_df, on="periodo", how="left"
    )
    generation_summary = generation_summary.merge(unclassified_df, on="periodo", how="left")
    for column in ["energia_termica_kwh", "energia_hidraulica_kwh", "energia_sin_clasificar_kwh"]:
        generation_summary[column] = pd.to_numeric(generation_summary[column], errors="coerce").fillna(0)

    generation_summary["participacion_termica_pct"] = (
        generation_summary["energia_termica_kwh"] / generation_summary["energia_total_kwh"] * 100
    ).where(generation_summary["energia_total_kwh"] > 0)
    generation_summary["participacion_hidraulica_pct"] = (
        generation_summary["energia_hidraulica_kwh"] / generation_summary["energia_total_kwh"] * 100
    ).where(generation_summary["energia_total_kwh"] > 0)
    generation_summary["cobertura_clasificada_pct"] = (
        (generation_summary["energia_total_kwh"] - generation_summary["energia_sin_clasificar_kwh"])
        / generation_summary["energia_total_kwh"]
        * 100
    ).where(generation_summary["energia_total_kwh"] > 0)

    emissions_base = emissions_df.copy()
    emissions_base["periodo"] = pd.to_datetime(emissions_base["periodo"], errors="coerce")
    emissions_base["emisiones_tco2"] = pd.to_numeric(emissions_base["emisiones_tco2"], errors="coerce")
    if "n_plantas" in emissions_base.columns:
        emissions_base["n_plantas"] = pd.to_numeric(emissions_base["n_plantas"], errors="coerce")
    emissions_base = emissions_base.dropna(subset=["periodo", "emisiones_tco2"])
    if emissions_base.empty:
        return empty_result

    summary = emissions_base.merge(generation_summary, on="periodo", how="outer")
    baseline_timestamp = pd.Timestamp(baseline_end_date)

    baseline_mask = (summary["periodo"] <= baseline_timestamp) & summary["periodo"].notna()
    if not baseline_mask.any():
        return empty_result

    baseline_share_series = summary.loc[baseline_mask, "participacion_termica_pct"].dropna()
    baseline_emissions_series = summary.loc[baseline_mask, "emisiones_tco2"].dropna()
    if baseline_share_series.empty or baseline_emissions_series.empty:
        return empty_result

    baseline_share_pct = float(baseline_share_series.mean())
    baseline_emissions_tco2 = float(baseline_emissions_series.mean())
    baseline_intensity_tco2_mwh = None

    thermal_excess_ratio = (
        (summary["participacion_termica_pct"] - baseline_share_pct).clip(lower=0)
        / summary["participacion_termica_pct"]
    ).where(summary["participacion_termica_pct"] > 0)
    summary["co2_adicional_ton"] = (summary["emisiones_tco2"] * thermal_excess_ratio).clip(lower=0)
    summary["emisiones_termicas_ton"] = summary["emisiones_tco2"]
    summary["energia_termica_adicional_kwh"] = (
        summary["energia_total_kwh"] * ((summary["participacion_termica_pct"] - baseline_share_pct).clip(lower=0) / 100)
    ).where(summary["energia_total_kwh"].notna())

    hydrology_available = False
    if not hydrology_df.empty:
        hydro_base = hydrology_df.copy()
        hydro_base["periodo"] = pd.to_datetime(hydro_base["periodo"], errors="coerce")
        hydro_base["ratio_hidrologico"] = pd.to_numeric(hydro_base["ratio_hidrologico"], errors="coerce")
        hydro_base = hydro_base.dropna(subset=["periodo", "ratio_hidrologico"])
        if not hydro_base.empty:
            summary = summary.merge(
                hydro_base[["periodo", "ratio_hidrologico"]],
                on="periodo",
                how="left",
            )
            hydrology_available = True

    summary = summary.sort_values("periodo").reset_index(drop=True)
    summary["participacion_termica_suavizada"] = summary["participacion_termica_pct"].rolling(2, min_periods=1).mean()
    summary["emisiones_suavizadas_ton"] = summary["emisiones_tco2"].rolling(2, min_periods=1).mean()

    trigger_mask = summary["periodo"] > baseline_timestamp
    if trigger_mask.any():
        trigger_mask &= summary["participacion_termica_pct"] >= (baseline_share_pct + 5)
        trigger_mask &= summary["emisiones_tco2"] >= (baseline_emissions_tco2 * 1.15)
        trigger_mask &= summary["co2_adicional_ton"].fillna(0) > 0
        if hydrology_available and "ratio_hidrologico" in summary.columns:
            trigger_mask &= summary["ratio_hidrologico"].fillna(1) < 0.95
        trigger_candidates = summary[trigger_mask]
        trigger_period = (
            pd.Timestamp(trigger_candidates.iloc[0]["periodo"])
            if not trigger_candidates.empty
            else None
        )
    else:
        trigger_period = None

    return {
        "summary": summary,
        "mix": mix_df,
        "drivers": drivers_df.copy(),
        "baseline_share_pct": baseline_share_pct,
        "baseline_emissions_tco2": baseline_emissions_tco2,
        "baseline_intensity_tco2_mwh": baseline_intensity_tco2_mwh,
        "trigger_period": trigger_period,
        "coverage_pct": float(summary["cobertura_clasificada_pct"].dropna().mean()),
        "hydrology_available": hydrology_available,
        "gold_source": True,
    }


def build_pareto_figure(
    dataframe: pd.DataFrame,
    category_column: str,
    value_column: str,
    title: str,
    bar_name: str,
) -> go.Figure:
    chart_df = dataframe.copy()
    chart_df[value_column] = pd.to_numeric(chart_df[value_column], errors="coerce")
    chart_df = chart_df.dropna(subset=[value_column]).sort_values(value_column, ascending=False)

    if chart_df.empty:
        return go.Figure()

    total_value = float(chart_df[value_column].sum())
    chart_df["participacion_pct"] = (
        (chart_df[value_column] / total_value) * 100 if total_value else 0
    )
    chart_df["participacion_acumulada"] = chart_df["participacion_pct"].cumsum()

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_bar(
        x=chart_df[category_column],
        y=chart_df[value_column],
        name=bar_name,
        marker_color="#1d4ed8",
        customdata=chart_df[["participacion_pct"]],
        hovertemplate="%{x}<br>Valor: %{y:,.0f}<br>Participacion: %{customdata[0]:.1f}%<extra></extra>",
    )
    figure.add_scatter(
        x=chart_df[category_column],
        y=chart_df["participacion_acumulada"],
        name="Participacion acumulada",
        mode="lines+markers",
        line=dict(color="#ea580c", width=3),
        marker=dict(size=8),
        hovertemplate="%{x}<br>Acumulado: %{y:.1f}%<extra></extra>",
        secondary_y=True,
    )
    figure.update_layout(
        title=title,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(tickangle=-30),
    )
    figure.update_yaxes(title_text=bar_name, secondary_y=False)
    figure.update_yaxes(title_text="Participacion acumulada (%)", range=[0, 105], secondary_y=True)
    return figure


def main() -> None:
    settings = load_settings()

    st.title("SIMEM Interactive Dashboards")
    st.caption(
        "Dashboard en Streamlit para explorar tablas silver y gold de SIMEM con consultas directas a Athena."
    )
    render_connection_status(settings)

    try:
        schemas = load_schemas(
            region_name=settings.region_name,
            workgroup=settings.workgroup,
            output_location=settings.output_location,
            database=settings.database,
        )
    except AthenaQueryError as error:
        st.error(
            "No pude consultar Athena para descubrir el schema de `simem_silver`. "
            f"Detalle: {error}"
        )
        st.stop()

    try:
        gold_schemas = load_schemas(
            region_name=settings.region_name,
            workgroup=settings.workgroup,
            output_location=settings.output_location,
            database=settings.gold_database,
            table_names=GOLD_TABLES,
        )
    except AthenaQueryError as error:
        st.warning(
            "No pude consultar Athena para descubrir el schema de `simem_gold`. "
            f"Seguire con los dashboards soportados por silver. Detalle: {error}"
        )
        gold_schemas = {}

    units_schema = schemas.get("unidades_generacion")
    generation_schema = schemas.get("generacion_real")
    hydrology_schema = schemas.get("aporte_hidricos")
    demand_real_schema = schemas.get("demanda_real")
    demand_commercial_schema = schemas.get("demanda_comercial")
    price_schema = schemas.get("precio_ponderado_bolsa")
    reservoir_schema = schemas.get("niveles_embalse")
    gold_emissions_daily_schema = gold_schemas.get("emisiones_sin_diario")
    gold_emissions_monthly_schema = gold_schemas.get("emisiones_sin_mensual")
    gold_generation_schema = gold_schemas.get("generacion")
    gold_hourly_features_schema = gold_schemas.get("demanda_real_hourly")
    gold_co2_plants_schema = gold_schemas.get("co2_tasa_planta_mes")
    gold_plants_schema = gold_schemas.get("plantas_catalogo")

    st.sidebar.header("Filtros")
    st.sidebar.caption("Los filtros se aplican a los visuales que tengan columnas compatibles.")

    generation_type_options: list[str] = []
    resource_state_options: list[str] = []
    region_options: list[str] = []
    market_type_options: list[str] = []

    if units_schema:
        generation_type_column = choose_column(units_schema, "generation_type")
        resource_state_column = choose_column(units_schema, "resource_state")
        if generation_type_column:
            generation_type_options = load_table_values(
                settings.region_name,
                settings.workgroup,
                settings.output_location,
                settings.database,
                units_schema.table_name,
                generation_type_column.name,
            )
        if resource_state_column:
            resource_state_options = load_table_values(
                settings.region_name,
                settings.workgroup,
                settings.output_location,
                settings.database,
                units_schema.table_name,
                resource_state_column.name,
            )

    if hydrology_schema:
        region_column = choose_column(hydrology_schema, "hydro_region")
        if region_column:
            region_options = load_table_values(
                settings.region_name,
                settings.workgroup,
                settings.output_location,
                settings.database,
                hydrology_schema.table_name,
                region_column.name,
            )

    for market_schema in (demand_real_schema, demand_commercial_schema):
        if not market_schema:
            continue
        market_type_column = choose_column(market_schema, "market_type")
        if not market_type_column:
            continue
        market_type_options.extend(
            load_table_values(
                settings.region_name,
                settings.workgroup,
                settings.output_location,
                settings.database,
                market_schema.table_name,
                market_type_column.name,
            )
        )
    market_type_options = sorted(set(market_type_options))

    default_states = [item for item in resource_state_options if item.lower() == "operacion"][:1]
    selected_generation_types = st.sidebar.multiselect(
        "Tipo de generacion",
        options=generation_type_options,
    )
    selected_resource_states = st.sidebar.multiselect(
        "Estado del recurso",
        options=resource_state_options,
        default=default_states,
    )
    selected_market_types = st.sidebar.multiselect(
        "Tipo de mercado",
        options=market_type_options,
    )
    selected_regions = st.sidebar.multiselect(
        "Region hidrologica",
        options=region_options,
    )

    detected_date_bounds: list[tuple[date | None, date | None]] = []
    for schema in (generation_schema, demand_real_schema, price_schema, hydrology_schema, reservoir_schema):
        if not schema:
            continue
        date_column = choose_column(schema, "date")
        if not date_column:
            continue
        detected_date_bounds.append(
            load_date_bounds(
                settings.region_name,
                settings.workgroup,
                settings.output_location,
                settings.database,
                schema.table_name,
                build_time_bucket(date_column),
            )
        )

    valid_starts = [item[0] for item in detected_date_bounds if item[0]]
    valid_ends = [item[1] for item in detected_date_bounds if item[1]]
    global_min = min(valid_starts) if valid_starts else None
    global_max = max(valid_ends) if valid_ends else None

    climate_detected_date_bounds: list[tuple[date | None, date | None]] = []
    for schema in (gold_emissions_daily_schema, gold_generation_schema, gold_hourly_features_schema):
        if not schema:
            continue
        date_column = schema.get_column("fechahora") or choose_column(schema, "date") or schema.get_column("event_ts")
        if not date_column:
            continue
        climate_detected_date_bounds.append(
            load_date_bounds(
                settings.region_name,
                settings.workgroup,
                settings.output_location,
                settings.gold_database,
                schema.table_name,
                build_time_bucket(date_column),
            )
        )

    climate_valid_starts = [item[0] for item in climate_detected_date_bounds if item[0]]
    climate_valid_ends = [item[1] for item in climate_detected_date_bounds if item[1]]
    climate_global_min = min(climate_valid_starts) if climate_valid_starts else global_min
    climate_global_max = max(climate_valid_ends) if climate_valid_ends else global_max

    if global_min and global_max:
        suggested_start = max(global_min, global_max - timedelta(days=90))
        selected_range = st.sidebar.date_input(
            "Rango de fechas",
            value=(suggested_start, global_max),
            min_value=global_min,
            max_value=global_max,
        )
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start_date, end_date = selected_range
        else:
            start_date, end_date = suggested_start, global_max
    else:
        start_date, end_date = None, None
        st.sidebar.info("No se detecto una columna de fecha comun para construir el filtro global.")

    previous_start_date, previous_end_date = _calculate_previous_period(start_date, end_date)

    event_default_start = date(2023, 1, 1)
    event_default_end = date(2024, 6, 30)
    if climate_global_min and climate_global_max:
        climate_default_start = _clamp_date(event_default_start, climate_global_min, climate_global_max)
        climate_default_end = _clamp_date(event_default_end, climate_global_min, climate_global_max)
        if climate_default_start > climate_default_end:
            climate_default_start, climate_default_end = climate_global_min, climate_global_max
    else:
        climate_default_start, climate_default_end = event_default_start, event_default_end

    climate_start_date, climate_end_date = _coerce_date_range(
        st.session_state.get("el_nino_event_range"),
        climate_default_start,
        climate_default_end,
    )
    climate_baseline_default = min(climate_end_date, date(2023, 8, 31))
    climate_baseline_end_date = st.session_state.get("el_nino_baseline_end", climate_baseline_default)
    if not isinstance(climate_baseline_end_date, date):
        climate_baseline_end_date = climate_baseline_default
    climate_baseline_end_date = _clamp_date(
        climate_baseline_end_date,
        climate_start_date,
        climate_end_date,
    )
    climate_granularity = st.session_state.get("el_nino_granularity", "Semanal")
    if climate_granularity not in {"Semanal", "Mensual"}:
        climate_granularity = "Semanal"

    summary_columns = st.columns(4)
    summary_columns[0].metric("Tablas silver detectadas", len(schemas))
    summary_columns[1].metric("Tablas gold detectadas", len(gold_schemas))
    summary_columns[2].metric("Tipos de generacion", len(generation_type_options))
    summary_columns[3].metric("Cobertura temporal", _format_date_bounds((global_min, global_max)))

    current_units_df = pd.DataFrame()
    current_demand_df = pd.DataFrame()
    previous_demand_df = pd.DataFrame()
    current_price_df = pd.DataFrame()
    previous_price_df = pd.DataFrame()
    current_duck_df = pd.DataFrame()
    previous_duck_df = pd.DataFrame()
    current_hydrology_df = pd.DataFrame()
    current_reservoir_df = pd.DataFrame()
    heatmap_df = pd.DataFrame()
    climate_emissions_df = pd.DataFrame()
    climate_generation_df = pd.DataFrame()
    climate_hydrology_df = pd.DataFrame()
    climate_drivers_df = pd.DataFrame()

    if units_schema:
        units_sql = build_units_distribution_query(
            database=settings.database,
            schema=units_schema,
            generation_types=selected_generation_types,
            resource_states=selected_resource_states,
        )
        if units_sql:
            current_units_df = load_dataframe(
                units_sql,
                region_name=settings.region_name,
                workgroup=settings.workgroup,
                output_location=settings.output_location,
            )
            if not current_units_df.empty:
                current_units_df["unidades"] = pd.to_numeric(current_units_df["unidades"], errors="coerce")

    current_demand_frames: list[pd.DataFrame] = []
    previous_demand_frames: list[pd.DataFrame] = []

    if demand_real_schema:
        current_real_sql = build_daily_series_query(
            settings.database,
            demand_real_schema,
            "real_demand",
            "Demanda real",
            start_date,
            end_date,
            selected_market_types,
            aggregation="SUM",
        )
        previous_real_sql = build_daily_series_query(
            settings.database,
            demand_real_schema,
            "real_demand",
            "Demanda real",
            previous_start_date,
            previous_end_date,
            selected_market_types,
            aggregation="SUM",
        )
        duck_sql = build_duck_curve_query(
            settings.database,
            demand_real_schema,
            start_date,
            end_date,
            selected_market_types,
        )
        previous_duck_sql = build_duck_curve_query(
            settings.database,
            demand_real_schema,
            previous_start_date,
            previous_end_date,
            selected_market_types,
        )
        heatmap_sql = build_hourly_heatmap_query(
            settings.database,
            demand_real_schema,
            start_date,
            end_date,
            selected_market_types,
        )

        if current_real_sql:
            current_demand_frames.append(
                load_dataframe(
                    current_real_sql,
                    region_name=settings.region_name,
                    workgroup=settings.workgroup,
                    output_location=settings.output_location,
                )
            )
        if previous_real_sql:
            previous_demand_frames.append(
                load_dataframe(
                    previous_real_sql,
                    region_name=settings.region_name,
                    workgroup=settings.workgroup,
                    output_location=settings.output_location,
                )
            )
        if duck_sql:
            current_duck_df = load_dataframe(
                duck_sql,
                region_name=settings.region_name,
                workgroup=settings.workgroup,
                output_location=settings.output_location,
            )
        if previous_duck_sql:
            previous_duck_df = load_dataframe(
                previous_duck_sql,
                region_name=settings.region_name,
                workgroup=settings.workgroup,
                output_location=settings.output_location,
            )
        if heatmap_sql:
            heatmap_df = load_dataframe(
                heatmap_sql,
                region_name=settings.region_name,
                workgroup=settings.workgroup,
                output_location=settings.output_location,
            )

    if demand_commercial_schema:
        current_commercial_sql = build_daily_series_query(
            settings.database,
            demand_commercial_schema,
            "commercial_demand",
            "Demanda comercial",
            start_date,
            end_date,
            selected_market_types,
            aggregation="SUM",
        )
        previous_commercial_sql = build_daily_series_query(
            settings.database,
            demand_commercial_schema,
            "commercial_demand",
            "Demanda comercial",
            previous_start_date,
            previous_end_date,
            selected_market_types,
            aggregation="SUM",
        )
        if current_commercial_sql:
            current_demand_frames.append(
                load_dataframe(
                    current_commercial_sql,
                    region_name=settings.region_name,
                    workgroup=settings.workgroup,
                    output_location=settings.output_location,
                )
            )
        if previous_commercial_sql:
            previous_demand_frames.append(
                load_dataframe(
                    previous_commercial_sql,
                    region_name=settings.region_name,
                    workgroup=settings.workgroup,
                    output_location=settings.output_location,
                )
            )

    if price_schema:
        current_price_sql = build_daily_series_query(
            settings.database,
            price_schema,
            "market_price",
            "Precio bolsa",
            start_date,
            end_date,
            aggregation="AVG",
        )
        previous_price_sql = build_daily_series_query(
            settings.database,
            price_schema,
            "market_price",
            "Precio bolsa",
            previous_start_date,
            previous_end_date,
            aggregation="AVG",
        )
        if current_price_sql:
            current_price_df = load_dataframe(
                current_price_sql,
                region_name=settings.region_name,
                workgroup=settings.workgroup,
                output_location=settings.output_location,
            )
        if previous_price_sql:
            previous_price_df = load_dataframe(
                previous_price_sql,
                region_name=settings.region_name,
                workgroup=settings.workgroup,
                output_location=settings.output_location,
            )

    if hydrology_schema:
        hydrology_sql = build_hydrology_query(
            database=settings.database,
            schema=hydrology_schema,
            start_date=start_date,
            end_date=end_date,
            regions=selected_regions,
        )
        if hydrology_sql:
            current_hydrology_df = load_dataframe(
                hydrology_sql,
                region_name=settings.region_name,
                workgroup=settings.workgroup,
                output_location=settings.output_location,
            )

    if reservoir_schema:
        reservoir_sql = build_reservoir_query(
            database=settings.database,
            schema=reservoir_schema,
            start_date=start_date,
            end_date=end_date,
        )
        if reservoir_sql:
            current_reservoir_df = load_dataframe(
                reservoir_sql,
                region_name=settings.region_name,
                workgroup=settings.workgroup,
                output_location=settings.output_location,
            )

    if climate_start_date and climate_end_date:
        climate_emissions_schema = (
            gold_emissions_monthly_schema if climate_granularity == "Mensual" else gold_emissions_daily_schema
        )
        if climate_emissions_schema:
            climate_emissions_sql = build_gold_emissions_query(
                database=settings.gold_database,
                schema=climate_emissions_schema,
                start_date=climate_start_date,
                end_date=climate_end_date,
                granularity=climate_granularity,
            )
            if climate_emissions_sql:
                climate_emissions_df = load_dataframe(
                    climate_emissions_sql,
                    region_name=settings.region_name,
                    workgroup=settings.workgroup,
                    output_location=settings.output_location,
                )

        if gold_generation_schema:
            climate_generation_sql = build_gold_generation_query(
                database=settings.gold_database,
                schema=gold_generation_schema,
                start_date=climate_start_date,
                end_date=climate_end_date,
                granularity=climate_granularity,
            )
            if climate_generation_sql:
                climate_generation_df = load_dataframe(
                    climate_generation_sql,
                    region_name=settings.region_name,
                    workgroup=settings.workgroup,
                    output_location=settings.output_location,
                )

        if gold_hourly_features_schema:
            climate_hydrology_sql = build_gold_hydrology_ratio_query(
                database=settings.gold_database,
                schema=gold_hourly_features_schema,
                start_date=climate_start_date,
                end_date=climate_end_date,
                granularity=climate_granularity,
            )
            if climate_hydrology_sql:
                climate_hydrology_df = load_dataframe(
                    climate_hydrology_sql,
                    region_name=settings.region_name,
                    workgroup=settings.workgroup,
                    output_location=settings.output_location,
                )

        drivers_start_date = max(climate_start_date, climate_baseline_end_date + timedelta(days=1))
        if gold_co2_plants_schema:
            climate_drivers_sql = build_gold_co2_drivers_query(
                database=settings.gold_database,
                co2_schema=gold_co2_plants_schema,
                plants_schema=gold_plants_schema,
                start_date=drivers_start_date,
                end_date=climate_end_date,
                limit=12,
            )
            if climate_drivers_sql:
                climate_drivers_df = load_dataframe(
                    climate_drivers_sql,
                    region_name=settings.region_name,
                    workgroup=settings.workgroup,
                    output_location=settings.output_location,
                )

    current_demand_df = _normalize_series(
        pd.concat(current_demand_frames, ignore_index=True) if current_demand_frames else pd.DataFrame()
    )
    previous_demand_df = _normalize_series(
        pd.concat(previous_demand_frames, ignore_index=True) if previous_demand_frames else pd.DataFrame()
    )
    current_price_df = _normalize_series(current_price_df)
    previous_price_df = _normalize_series(previous_price_df)
    units_breakdown_df = prepare_units_breakdown(current_units_df)

    if not current_duck_df.empty:
        current_duck_df["hora"] = pd.to_numeric(current_duck_df["hora"], errors="coerce")
        current_duck_df["demanda_promedio"] = pd.to_numeric(
            current_duck_df["demanda_promedio"], errors="coerce"
        )
        current_duck_df["demanda_minima"] = pd.to_numeric(
            current_duck_df["demanda_minima"], errors="coerce"
        )
        current_duck_df["demanda_maxima"] = pd.to_numeric(
            current_duck_df["demanda_maxima"], errors="coerce"
        )

    if not previous_duck_df.empty:
        previous_duck_df["demanda_promedio"] = pd.to_numeric(
            previous_duck_df["demanda_promedio"], errors="coerce"
        )

    if not heatmap_df.empty:
        heatmap_df["dia_num"] = pd.to_numeric(heatmap_df["dia_num"], errors="coerce")
        heatmap_df["hora"] = pd.to_numeric(heatmap_df["hora"], errors="coerce")
        heatmap_df["demanda_promedio"] = pd.to_numeric(heatmap_df["demanda_promedio"], errors="coerce")

    if not current_hydrology_df.empty:
        current_hydrology_df["periodo"] = pd.to_datetime(current_hydrology_df["periodo"], errors="coerce")
        current_hydrology_df["aporte_actual"] = pd.to_numeric(
            current_hydrology_df["aporte_actual"], errors="coerce"
        )
        if "media_historica" in current_hydrology_df.columns:
            current_hydrology_df["media_historica"] = pd.to_numeric(
                current_hydrology_df["media_historica"], errors="coerce"
            )

    if not current_reservoir_df.empty:
        current_reservoir_df["periodo"] = pd.to_datetime(current_reservoir_df["periodo"], errors="coerce")
        current_reservoir_df["nivel"] = pd.to_numeric(current_reservoir_df["nivel"], errors="coerce")

    if not climate_emissions_df.empty:
        climate_emissions_df["periodo"] = pd.to_datetime(climate_emissions_df["periodo"], errors="coerce")
        climate_emissions_df["emisiones_tco2"] = pd.to_numeric(
            climate_emissions_df["emisiones_tco2"], errors="coerce"
        )
        if "n_plantas" in climate_emissions_df.columns:
            climate_emissions_df["n_plantas"] = pd.to_numeric(
                climate_emissions_df["n_plantas"], errors="coerce"
            )

    if not climate_generation_df.empty:
        climate_generation_df["periodo"] = pd.to_datetime(climate_generation_df["periodo"], errors="coerce")
        climate_generation_df["energia_kwh"] = pd.to_numeric(
            climate_generation_df["energia_kwh"], errors="coerce"
        )

    if not climate_hydrology_df.empty:
        climate_hydrology_df["periodo"] = pd.to_datetime(climate_hydrology_df["periodo"], errors="coerce")
        climate_hydrology_df["ratio_hidrologico"] = pd.to_numeric(
            climate_hydrology_df["ratio_hidrologico"], errors="coerce"
        )

    if not climate_drivers_df.empty:
        for numeric_column in ("emisiones_tco2", "generacion_mwh", "tasa_tco2_mwh"):
            if numeric_column in climate_drivers_df.columns:
                climate_drivers_df[numeric_column] = pd.to_numeric(
                    climate_drivers_df[numeric_column], errors="coerce"
                )

    current_real_total = _extract_series_metric(current_demand_df, "Demanda real", "sum")
    previous_real_total = _extract_series_metric(previous_demand_df, "Demanda real", "sum")
    current_commercial_total = _extract_series_metric(current_demand_df, "Demanda comercial", "sum")
    previous_commercial_total = _extract_series_metric(previous_demand_df, "Demanda comercial", "sum")
    current_price_avg = _extract_series_metric(current_price_df, "Precio bolsa", "mean")
    previous_price_avg = _extract_series_metric(previous_price_df, "Precio bolsa", "mean")
    current_peak_hour = (
        float(current_duck_df["demanda_promedio"].max())
        if not current_duck_df.empty and current_duck_df["demanda_promedio"].notna().any()
        else None
    )
    previous_peak_hour = (
        float(previous_duck_df["demanda_promedio"].max())
        if not previous_duck_df.empty and previous_duck_df["demanda_promedio"].notna().any()
        else None
    )
    current_operating_units = (
        float(current_units_df["unidades"].sum())
        if not current_units_df.empty and current_units_df["unidades"].notna().any()
        else None
    )
    latest_reservoir_mean = None
    if not current_reservoir_df.empty:
        latest_mask = current_reservoir_df["es_ultimo_corte"].astype(str).str.lower() == "true"
        latest_df = current_reservoir_df[latest_mask]
        if not latest_df.empty and latest_df["nivel"].notna().any():
            latest_reservoir_mean = float(latest_df["nivel"].mean())

    comparison_df = build_comparison_dataframe(
        current_demand_df,
        previous_demand_df,
        current_price_df,
        previous_price_df,
    )

    el_nino_analysis = prepare_el_nino_analysis(
        climate_emissions_df,
        climate_generation_df,
        climate_hydrology_df,
        climate_drivers_df,
        climate_baseline_end_date,
        climate_granularity,
    )

    executive_tab, climate_tab, units_tab, market_tab, hydrology_tab, schema_tab = st.tabs(
        [
            "Resumen Ejecutivo",
            "El Nino y CO2",
            "Oferta y Unidades",
            "Demanda y Mercado",
            "Agua y Embalses",
            "Explorador de Tablas",
        ]
    )

    with executive_tab:
        st.subheader("KPIs del periodo seleccionado")
        st.caption(
            f"Periodo actual: {_format_range_label(start_date, end_date)} | "
            f"Periodo anterior: {_format_range_label(previous_start_date, previous_end_date)}"
        )

        metric_columns = st.columns(3)
        metric_columns[0].metric(
            "Demanda real total",
            _format_number(current_real_total, decimals=0),
            _delta_label(current_real_total, previous_real_total),
        )
        metric_columns[1].metric(
            "Demanda comercial total",
            _format_number(current_commercial_total, decimals=0),
            _delta_label(current_commercial_total, previous_commercial_total),
        )
        metric_columns[2].metric(
            "Precio promedio bolsa",
            _format_number(current_price_avg, decimals=1, suffix=" COP/kWh"),
            _delta_label(current_price_avg, previous_price_avg),
        )

        metric_columns = st.columns(3)
        metric_columns[0].metric(
            "Pico horario promedio",
            _format_number(current_peak_hour, decimals=0),
            _delta_label(current_peak_hour, previous_peak_hour),
        )
        metric_columns[1].metric(
            "Unidades en operacion",
            _format_number(current_operating_units, decimals=0),
        )
        metric_columns[2].metric(
            "Nivel promedio embalses",
            _format_number(latest_reservoir_mean, decimals=1, suffix="%"),
        )

        st.markdown("### Comparativo de periodos")
        if not comparison_df.empty:
            comparison_chart_df = comparison_df.melt(
                id_vars="metrica",
                value_vars=["periodo_actual", "periodo_anterior"],
                var_name="periodo",
                value_name="valor",
            )
            comparison_chart_df["periodo"] = comparison_chart_df["periodo"].map(
                {
                    "periodo_actual": "Periodo actual",
                    "periodo_anterior": "Periodo anterior",
                }
            )
            comparison_chart = px.bar(
                comparison_chart_df,
                x="metrica",
                y="valor",
                color="periodo",
                barmode="group",
                title="Actual vs anterior",
            )
            comparison_chart.update_layout(hovermode="x unified")
            st.plotly_chart(comparison_chart, width="stretch")

            display_df = comparison_df.copy()
            display_df["periodo_actual"] = display_df["periodo_actual"].map(lambda value: _format_number(value, 1))
            display_df["periodo_anterior"] = display_df["periodo_anterior"].map(
                lambda value: _format_number(value, 1)
            )
            display_df["variacion_pct"] = display_df["variacion_pct"].map(
                lambda value: "N/D" if value is None or pd.isna(value) else f"{value:+.1f}%"
            )
            st.dataframe(
                display_df.rename(
                    columns={
                        "metrica": "Metrica",
                        "periodo_actual": "Periodo actual",
                        "periodo_anterior": "Periodo anterior",
                        "variacion_pct": "Variacion %",
                    }
                ),
                width="stretch",
            )

        st.markdown("### Panorama del sistema")
        overview_left, overview_right = st.columns(2)
        if not units_breakdown_df.empty:
            executive_units_metric = overview_left.radio(
                "Ver matriz operativa en",
                options=["Participacion %", "Unidades"],
                horizontal=True,
                key="executive_units_metric",
            )
            units_x_field = "participacion_pct" if executive_units_metric == "Participacion %" else "unidades"
            units_text_field = (
                "participacion_label" if executive_units_metric == "Participacion %" else "unidades_label"
            )
            units_chart = px.bar(
                units_breakdown_df.sort_values(units_x_field, ascending=True),
                x=units_x_field,
                y="tipo_generacion",
                orientation="h",
                color="participacion_pct",
                color_continuous_scale="Blues",
                text=units_text_field,
                title="Composicion operativa por tipo de generacion",
                labels={
                    "tipo_generacion": "Tipo de generacion",
                    "participacion_pct": "Participacion (%)",
                    "unidades": "Unidades",
                },
            )
            units_chart.update_layout(coloraxis_showscale=False)
            overview_left.plotly_chart(units_chart, width="stretch")
        else:
            overview_left.info("No hay datos de unidades para el filtro seleccionado.")

        if not current_demand_df.empty:
            executive_series_options = sorted(current_demand_df["serie"].dropna().unique().tolist())
            executive_selected_series = overview_right.multiselect(
                "Series de demanda",
                options=executive_series_options,
                default=executive_series_options,
                key="executive_demand_series",
            )
            executive_demand_df = current_demand_df[
                current_demand_df["serie"].isin(executive_selected_series)
            ].copy()
            if not executive_demand_df.empty:
                line = px.line(
                    executive_demand_df,
                    x="periodo",
                    y="valor",
                    color="serie",
                    markers=True,
                    title="Demanda diaria",
                    labels={"valor": "Demanda"},
                )
                line.update_xaxes(rangeslider_visible=True)
                overview_right.plotly_chart(line, width="stretch")
            else:
                overview_right.info("Selecciona al menos una serie de demanda para el resumen.")
        else:
            overview_right.info("No hay datos de demanda disponibles para el periodo elegido.")

        st.markdown("### Preguntas de negocio")
        st.markdown(
            "- Como esta compuesta la matriz operativa por tipo de generacion.\n"
            "- Como viene evolucionando la demanda real frente a la comercial.\n"
            "- Si el precio ponderado de bolsa acompana los cambios de demanda.\n"
            "- Que regiones o embalses merecen seguimiento mas cercano.\n"
            "- En que momento El Nino 2023-2024 elevo la dependencia termica y el costo estimado en CO2."
        )

    with climate_tab:
        st.subheader("El Nino 2023-2024: dependencia termica y costo en CO2")
        st.caption(
            "Esta vista usa tablas `gold`: `emisiones_sin_diario|mensual` para el CO2 real del Sistema Interconectado Nacional, "
            "`generacion` para el mix por tecnologia, `co2_tasa_planta_mes` para los emisores lideres "
            "y `demanda_real_hourly` para el ratio hidrologico vs media."
        )

        climate_controls = st.columns(4)
        climate_min_date = climate_global_min or climate_start_date
        climate_max_date = climate_global_max or climate_end_date
        climate_controls[0].date_input(
            "Ventana del evento",
            value=(climate_start_date, climate_end_date),
            min_value=climate_min_date,
            max_value=climate_max_date,
            key="el_nino_event_range",
        )
        climate_controls[1].date_input(
            "Fin del baseline preevento",
            value=climate_baseline_end_date,
            min_value=climate_start_date,
            max_value=climate_end_date,
            key="el_nino_baseline_end",
        )
        climate_controls[2].selectbox(
            "Granularidad",
            options=["Semanal", "Mensual"],
            key="el_nino_granularity",
        )
        climate_controls[3].selectbox(
            "Desglose de emisores",
            options=["Plantas", "Combustibles"],
            key="el_nino_driver_view",
        )

        climate_summary_df = el_nino_analysis["summary"]
        climate_mix_df = el_nino_analysis["mix"]
        climate_drivers_df = el_nino_analysis["drivers"]
        trigger_period = el_nino_analysis["trigger_period"]
        baseline_share_pct = el_nino_analysis["baseline_share_pct"]
        baseline_emissions_tco2 = el_nino_analysis["baseline_emissions_tco2"]
        coverage_pct = el_nino_analysis["coverage_pct"]
        hydrology_available = bool(el_nino_analysis["hydrology_available"])
        driver_view = st.session_state.get("el_nino_driver_view", "Plantas")

        if coverage_pct is not None and not pd.isna(coverage_pct) and coverage_pct < 90:
            st.warning(
                f"La clasificacion del mix cubre aproximadamente {_format_number(coverage_pct, 1, '%')} "
                "de la energia observada. La lectura del share termico puede quedar subestimada en los periodos con huecos."
            )

        if climate_summary_df.empty or climate_mix_df.empty:
            st.info(
                "No encontre suficientes datos para responder esta pregunta. "
                "Necesito al menos `emisiones_sin_*` y `generacion` dentro de `simem_gold`."
            )
        else:
            baseline_timestamp = pd.Timestamp(climate_baseline_end_date)
            post_baseline_df = climate_summary_df[climate_summary_df["periodo"] > baseline_timestamp].copy()
            missing_mix_periods = climate_summary_df[
                climate_summary_df["emisiones_tco2"].notna()
                & climate_summary_df["participacion_termica_pct"].isna()
            ]["periodo"].dropna()
            if not missing_mix_periods.empty:
                missing_labels = ", ".join(
                    missing_mix_periods.head(4).map(lambda value: _format_period_label(value, climate_granularity))
                )
                st.info(
                    "Hay periodos con emisiones del Sistema Interconectado Nacional pero sin detalle de mix en `gold.generacion`: "
                    f"{missing_labels}. Esos cortes salen en la serie de CO2, pero no en el calculo del share termico."
                )

            peak_share_row = (
                climate_summary_df.loc[climate_summary_df["participacion_termica_pct"].idxmax()]
                if climate_summary_df["participacion_termica_pct"].notna().any()
                else None
            )
            peak_emissions_row = (
                climate_summary_df.loc[climate_summary_df["emisiones_tco2"].idxmax()]
                if climate_summary_df["emisiones_tco2"].notna().any()
                else None
            )
            peak_additional_row = (
                post_baseline_df.loc[post_baseline_df["co2_adicional_ton"].idxmax()]
                if not post_baseline_df.empty and post_baseline_df["co2_adicional_ton"].notna().any()
                else None
            )

            additional_co2_total = (
                float(post_baseline_df["co2_adicional_ton"].sum())
                if not post_baseline_df.empty and post_baseline_df["co2_adicional_ton"].notna().any()
                else None
            )
            event_co2_total = (
                float(post_baseline_df["emisiones_tco2"].sum())
                if not post_baseline_df.empty and post_baseline_df["emisiones_tco2"].notna().any()
                else None
            )
            additional_share_of_event = (
                (additional_co2_total / event_co2_total) * 100
                if additional_co2_total is not None
                and event_co2_total not in (None, 0)
                and not pd.isna(additional_co2_total)
                and not pd.isna(event_co2_total)
                else None
            )

            trigger_label = _format_period_label(trigger_period, climate_granularity)
            trigger_delta = None
            if trigger_period is not None:
                trigger_point = climate_summary_df[climate_summary_df["periodo"] == trigger_period]
                if not trigger_point.empty and hydrology_available and "ratio_hidrologico" in trigger_point.columns:
                    trigger_ratio = trigger_point.iloc[0]["ratio_hidrologico"]
                    if pd.notna(trigger_ratio):
                        trigger_gap_pct = abs(1 - float(trigger_ratio)) * 100
                        if float(trigger_ratio) < 1:
                            trigger_delta = (
                                f"{_format_number(trigger_gap_pct, 0, '%')} por debajo de la media hidrologica"
                            )
                        else:
                            trigger_delta = (
                                f"{_format_number(trigger_gap_pct, 0, '%')} por encima de la media hidrologica"
                            )
            event_highlight_start, event_highlight_end = _resolve_highlight_window(
                trigger_period=trigger_period,
                fallback_start=climate_start_date,
                event_end=climate_end_date,
            )

            metric_columns = st.columns(4)
            metric_columns[0].metric(
                "Momento del disparo",
                trigger_label if trigger_period is not None else "No detectado",
                trigger_delta,
            )
            metric_columns[1].metric(
                "Pico de participacion termica",
                (
                    _format_number(float(peak_share_row["participacion_termica_pct"]), 1, "%")
                    if peak_share_row is not None
                    else "N/D"
                ),
                (
                    f"{abs(float(peak_share_row['participacion_termica_pct']) - float(baseline_share_pct)):.1f} pp "
                    + (
                        "por encima del nivel preevento"
                        if float(peak_share_row["participacion_termica_pct"]) >= float(baseline_share_pct)
                        else "por debajo del nivel preevento"
                    )
                    if peak_share_row is not None
                    and baseline_share_pct is not None
                    and not pd.isna(baseline_share_pct)
                    else None
                ),
            )
            metric_columns[2].metric(
                "CO2 adicional del mix",
                _format_number(additional_co2_total, 0, " tCO2"),
                (
                    _format_number(additional_share_of_event, 1, "%") + " del CO2 del periodo fue adicional"
                    if additional_share_of_event is not None and not pd.isna(additional_share_of_event)
                    else None
                ),
            )
            metric_columns[3].metric(
                "Pico de emisiones del Sistema Interconectado Nacional",
                (
                    _format_number(float(peak_emissions_row["emisiones_tco2"]), 0, " tCO2")
                    if peak_emissions_row is not None
                    else "N/D"
                ),
                (
                    "Pico en " + _format_period_label(peak_emissions_row["periodo"], climate_granularity)
                    if peak_emissions_row is not None
                    else None
                ),
            )

            insights = []
            if trigger_period is not None:
                insights.append(
                    f"- El quiebre aparece en `{_format_period_label(trigger_period, climate_granularity)}`, "
                    f"cuando la participacion termica suavizada supera el baseline de {_format_number(baseline_share_pct, 1, '%')} "
                    f"y el CO2 del Sistema Interconectado Nacional supera el baseline de {_format_number(baseline_emissions_tco2, 0, ' tCO2')} por periodo."
                )
            if peak_additional_row is not None:
                insights.append(
                    f"- El mayor sobrecosto de CO2 llega en `{_format_period_label(peak_additional_row['periodo'], climate_granularity)}` "
                    f"con {_format_number(float(peak_additional_row['co2_adicional_ton']), 0, ' tCO2')} adicionales."
                )
            if peak_emissions_row is not None:
                insights.append(
                    f"- El pico absoluto de emisiones del Sistema Interconectado Nacional ocurre en `{_format_period_label(peak_emissions_row['periodo'], climate_granularity)}` "
                    f"con {_format_number(float(peak_emissions_row['emisiones_tco2']), 0, ' tCO2')}."
                )
            if hydrology_available and "ratio_hidrologico" in climate_summary_df.columns:
                lowest_ratio_df = climate_summary_df.dropna(subset=["ratio_hidrologico"])
                if not lowest_ratio_df.empty:
                    stress_row = lowest_ratio_df.sort_values("ratio_hidrologico").iloc[0]
                    insights.append(
                        f"- El estres hidrologico mas fuerte del recorte ocurre en `{_format_period_label(stress_row['periodo'], climate_granularity)}`, "
                        f"con aportes en {_format_number(float(stress_row['ratio_hidrologico']) * 100, 0, '%')} de la media."
                    )

            if insights:
                st.markdown("### Lectura ejecutiva")
                st.markdown("\n".join(insights))
            if event_highlight_start is not None and event_highlight_end is not None:
                st.caption(
                    "Ventana resaltada en las graficas: "
                    f"`{_format_period_label(event_highlight_start, climate_granularity)}` -> "
                    f"`{_format_period_label(event_highlight_end, climate_granularity)}`. "
                    "El inicio usa el primer quiebre detectado y el cierre el fin del rango analizado."
                )

            climate_left, climate_right = st.columns(2)

            emissions_chart = go.Figure()
            _add_time_window_highlight(emissions_chart, event_highlight_start, event_highlight_end)
            emissions_chart.add_bar(
                x=climate_summary_df["periodo"],
                y=climate_summary_df["co2_adicional_ton"],
                name="CO2 adicional vs baseline",
                marker_color="#c2410c",
                hovertemplate="%{x|%Y-%m-%d}<br>CO2 adicional: %{y:,.0f} t<extra></extra>",
            )
            emissions_chart.add_scatter(
                x=climate_summary_df["periodo"],
                y=climate_summary_df["emisiones_tco2"],
                name="CO2 real del sistema",
                mode="lines",
                line=dict(color="#1d4ed8", width=3),
                hovertemplate="%{x|%Y-%m-%d}<br>CO2 Sistema Interconectado Nacional: %{y:,.0f} t<extra></extra>",
            )
            if baseline_emissions_tco2 is not None and not pd.isna(baseline_emissions_tco2):
                emissions_chart.add_hline(
                    y=float(baseline_emissions_tco2),
                    line_dash="dash",
                    line_color="#64748b",
                    annotation_text="Baseline preevento",
                    annotation_position="top left",
                    annotation_font_color="#64748b",
                    annotation_font_size=15,
                    annotation_bgcolor="rgba(248, 250, 252, 0.88)",
                    annotation_bordercolor="#94a3b8",
                    annotation_borderpad=4,
                )
            if trigger_period is not None:
                trigger_y_max = float(
                    max(
                        climate_summary_df["emisiones_tco2"].max(),
                        climate_summary_df["co2_adicional_ton"].max(),
                    )
                )
                emissions_chart.add_scatter(
                    x=[trigger_period, trigger_period],
                    y=[0, trigger_y_max],
                    name="Disparo",
                    mode="lines",
                    line=dict(color="#dc2626", width=2, dash="dot"),
                    hovertemplate="%{x|%Y-%m-%d}<extra>Disparo</extra>",
                    showlegend=False,
                )
                emissions_chart.add_annotation(
                    x=trigger_period,
                    y=trigger_y_max,
                    text="Disparo",
                    showarrow=False,
                    yshift=12,
                    font=dict(color="#dc2626"),
                )
            emissions_chart.update_layout(
                title="Emisiones del Sistema Interconectado Nacional",
                barmode="overlay",
                hovermode="x unified",
                yaxis_title="tCO2 por periodo",
                xaxis_title="Periodo",
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                margin=dict(r=170, b=40),
            )
            climate_left.plotly_chart(emissions_chart, width="stretch")

            dependency_chart = make_subplots(specs=[[{"secondary_y": True}]])
            _add_time_window_highlight(dependency_chart, event_highlight_start, event_highlight_end)
            dependency_chart.add_scatter(
                x=climate_summary_df["periodo"],
                y=climate_summary_df["participacion_termica_pct"],
                name="Participacion termica",
                mode="lines",
                line=dict(color="#b91c1c", width=3),
                secondary_y=False,
            )
            dependency_chart.add_scatter(
                x=climate_summary_df["periodo"],
                y=climate_summary_df["participacion_hidraulica_pct"],
                name="Participacion hidraulica",
                mode="lines",
                line=dict(color="#0f766e", width=2),
                secondary_y=False,
            )
            if baseline_share_pct is not None and not pd.isna(baseline_share_pct):
                dependency_chart.add_scatter(
                    x=climate_summary_df["periodo"],
                    y=[float(baseline_share_pct)] * len(climate_summary_df),
                    name="Baseline termico",
                    mode="lines",
                    line=dict(color="#7f1d1d", width=1, dash="dash"),
                    secondary_y=False,
                )
            if hydrology_available and "ratio_hidrologico" in climate_summary_df.columns:
                dependency_chart.add_scatter(
                    x=climate_summary_df["periodo"],
                    y=climate_summary_df["ratio_hidrologico"] * 100,
                    name="Aportes / media historica",
                    mode="lines",
                    line=dict(color="#0369a1", width=2, dash="dot"),
                    secondary_y=True,
                )
                dependency_chart.update_yaxes(
                    title_text="",
                    showticklabels=False,
                    secondary_y=True,
                )
            dependency_chart.update_layout(
                title="Dependencia termica frente al pulso hidrologico",
                hovermode="x unified",
            )
            dependency_chart.update_yaxes(title_text="Participacion en la energia (%)", secondary_y=False)
            climate_right.plotly_chart(dependency_chart, width="stretch")

            lower_left, lower_right = st.columns(2)
            mix_chart_df = climate_mix_df.copy()
            mix_chart_df["energia_gwh"] = mix_chart_df["energia_kwh"] / 1_000_000
            mix_chart = px.area(
                mix_chart_df,
                x="periodo",
                y="energia_gwh",
                color="bucket",
                category_orders={"bucket": GENERATION_BUCKET_ORDER},
                title="Mix de generacion del Sistema Interconectado Nacional",
                labels={"energia_gwh": "Energia (GWh)", "bucket": "Tecnologia"},
            )
            _add_time_window_highlight(mix_chart, event_highlight_start, event_highlight_end)
            lower_left.plotly_chart(mix_chart, width="stretch")

            if not climate_drivers_df.empty:
                if driver_view == "Combustibles":
                    fuel_df = (
                        climate_drivers_df.groupby("combustible_dom", as_index=False)
                        .agg(
                            emisiones_tco2=("emisiones_tco2", "sum"),
                            generacion_mwh=("generacion_mwh", "sum"),
                        )
                        .sort_values("emisiones_tco2", ascending=False)
                    )
                    fuel_chart = px.bar(
                        fuel_df.head(10).sort_values("emisiones_tco2", ascending=True),
                        x="emisiones_tco2",
                        y="combustible_dom",
                        orientation="h",
                        title="Combustibles que mas empujaron el CO2",
                        labels={"emisiones_tco2": "tCO2", "combustible_dom": "Combustible"},
                        color="emisiones_tco2",
                        color_continuous_scale="OrRd",
                    )
                    fuel_chart.update_layout(coloraxis_showscale=False)
                    lower_right.plotly_chart(fuel_chart, width="stretch")
                    lower_right.dataframe(
                        fuel_df.rename(
                            columns={
                                "combustible_dom": "Combustible",
                                "emisiones_tco2": "Emisiones (tCO2)",
                                "generacion_mwh": "Generacion (MWh)",
                            }
                        ).style.format({"Emisiones (tCO2)": "{:,.0f}", "Generacion (MWh)": "{:,.0f}"}),
                        width="stretch",
                    )
                else:
                    plant_chart_df = climate_drivers_df.head(10).copy()
                    plant_chart = px.bar(
                        plant_chart_df.sort_values("emisiones_tco2", ascending=True),
                        x="emisiones_tco2",
                        y="planta_nombre",
                        orientation="h",
                        color="combustible_dom",
                        title="Plantas que mas elevaron el CO2 del evento",
                        labels={"emisiones_tco2": "tCO2", "planta_nombre": "Planta"},
                        hover_data={"codigo_planta": True, "generacion_mwh": ":,.0f", "tasa_tco2_mwh": ":.3f"},
                    )
                    lower_right.plotly_chart(plant_chart, width="stretch")
            else:
                ranking_df = climate_summary_df.nlargest(8, "co2_adicional_ton")[
                    ["periodo", "participacion_termica_pct", "co2_adicional_ton", "emisiones_tco2"]
                ].copy()
                ranking_df["periodo"] = ranking_df["periodo"].map(
                    lambda value: _format_period_label(value, climate_granularity)
                )
                ranking_df = ranking_df.rename(
                    columns={
                        "periodo": "Periodo",
                        "participacion_termica_pct": "Participacion termica (%)",
                        "co2_adicional_ton": "CO2 adicional (t)",
                        "emisiones_tco2": "CO2 Sistema Interconectado Nacional (t)",
                    }
                )
                lower_right.dataframe(
                    ranking_df.style.format(
                        {
                            "Participacion termica (%)": "{:.1f}",
                            "CO2 adicional (t)": "{:,.0f}",
                            "CO2 Sistema Interconectado Nacional (t)": "{:,.0f}",
                        }
                    ),
                    width="stretch",
                )

            with st.expander("Metodologia de la estimacion", expanded=False):
                st.markdown(
                    "1. `emisiones_sin_diario` o `emisiones_sin_mensual` aporta el CO2 real del Sistema Interconectado Nacional por periodo.\n"
                    "2. `generacion` aporta el mix de energia por tecnologia y permite medir la participacion termica.\n"
                    "3. El `baseline` es el promedio hasta la fecha de corte preevento que selecciones.\n"
                    "4. El `CO2 adicional del mix` atribuye a la dependencia termica la fraccion del CO2 observada que excede la participacion termica promedio del baseline.\n"
                    "5. En terminos simples: si el share termico sube frente al baseline, se imputa como costo incremental la porcion equivalente del CO2 real del Sistema Interconectado Nacional.\n"
                    "6. `co2_tasa_planta_mes` se usa para identificar que plantas o combustibles explican el sobrecosto en CO2."
                )

    with units_tab:
        st.subheader("Oferta operativa")
        st.caption("Sin graficas de torta: aqui la composicion se entiende mejor con barras y concentracion acumulada.")

        units_controls = st.columns(3)
        units_metric_view = units_controls[0].radio(
            "Metrica de composicion",
            options=["Unidades", "Participacion %"],
            horizontal=True,
            key="units_metric_view",
        )
        top_plants_limit = units_controls[1].slider(
            "Top plantas a mostrar",
            min_value=5,
            max_value=20,
            value=10,
            key="units_top_plants_limit",
        )
        show_units_table = units_controls[2].checkbox(
            "Ver tabla resumen",
            value=True,
            key="show_units_table",
        )

        if units_schema:
            top_plants_sql = build_units_top_plants_query(
                database=settings.database,
                schema=units_schema,
                generation_types=selected_generation_types,
                resource_states=selected_resource_states,
                limit=top_plants_limit,
            )

            col_left, col_right = st.columns(2)
            if not units_breakdown_df.empty:
                units_x_field = "participacion_pct" if units_metric_view == "Participacion %" else "unidades"
                units_text_field = (
                    "participacion_label" if units_metric_view == "Participacion %" else "unidades_label"
                )
                composition_chart = px.bar(
                    units_breakdown_df.sort_values(units_x_field, ascending=True),
                    x=units_x_field,
                    y="tipo_generacion",
                    orientation="h",
                    color="participacion_pct",
                    color_continuous_scale="Tealgrn",
                    text=units_text_field,
                    title="Composicion de unidades en operacion",
                    labels={
                        "tipo_generacion": "Tipo de generacion",
                        "participacion_pct": "Participacion (%)",
                        "unidades": "Unidades",
                    },
                )
                composition_chart.update_layout(coloraxis_showscale=False)
                col_left.plotly_chart(composition_chart, width="stretch")
            else:
                col_left.info("No hay unidades operativas con los filtros seleccionados.")

            if top_plants_sql:
                top_df = load_dataframe(
                    top_plants_sql,
                    region_name=settings.region_name,
                    workgroup=settings.workgroup,
                    output_location=settings.output_location,
                )
                if not top_df.empty:
                    top_df["unidades"] = pd.to_numeric(top_df["unidades"], errors="coerce")
                    pareto_chart = build_pareto_figure(
                        top_df,
                        category_column="planta",
                        value_column="unidades",
                        title=f"Concentracion de unidades por planta (top {top_plants_limit})",
                        bar_name="Unidades",
                    )
                    col_right.plotly_chart(pareto_chart, width="stretch")
                else:
                    col_right.info("No se encontraron plantas para construir el ranking.")

        if show_units_table and not units_breakdown_df.empty:
            units_table_df = units_breakdown_df.rename(
                columns={
                    "tipo_generacion": "Tipo de generacion",
                    "unidades": "Unidades",
                    "participacion_pct": "Participacion (%)",
                }
            )[["Tipo de generacion", "Unidades", "Participacion (%)"]]
            st.dataframe(
                units_table_df.style.format({"Unidades": "{:,.0f}", "Participacion (%)": "{:.1f}%"}),
                width="stretch",
            )

    with market_tab:
        st.subheader("Demanda y mercado")
        st.caption(
            f"Comparando {_format_range_label(start_date, end_date)} contra "
            f"{_format_range_label(previous_start_date, previous_end_date)}"
        )

        market_controls = st.columns(3)
        demand_view = market_controls[0].selectbox(
            "Vista de demanda",
            options=["Diaria", "Promedio movil", "Acumulada"],
            key="market_demand_view",
        )
        available_demand_series = (
            sorted(current_demand_df["serie"].dropna().unique().tolist())
            if "serie" in current_demand_df.columns
            else []
        )
        selected_demand_series = market_controls[1].multiselect(
            "Series de demanda",
            options=available_demand_series,
            default=available_demand_series,
            key="market_selected_series",
        )
        market_analysis_mode = market_controls[2].selectbox(
            "Analisis complementario",
            options=["Precio diario", "Series indexadas", "Dispersion demanda-precio"],
            key="market_analysis_mode",
        )

        rolling_window = 7
        if demand_view == "Promedio movil":
            rolling_window = st.slider(
                "Ventana del promedio movil (dias)",
                min_value=3,
                max_value=21,
                value=7,
                step=2,
                key="market_rolling_window",
            )

        market_left, market_right = st.columns(2)
        if not current_demand_df.empty and selected_demand_series:
            demand_view_df = current_demand_df[
                current_demand_df["serie"].isin(selected_demand_series)
            ].copy()
            demand_view_df = demand_view_df.sort_values(["serie", "periodo"])
            demand_title = "Demanda diaria real vs comercial"
            if demand_view == "Promedio movil":
                demand_view_df["valor"] = demand_view_df.groupby("serie")["valor"].transform(
                    lambda values: values.rolling(rolling_window, min_periods=1).mean()
                )
                demand_title = f"Demanda con promedio movil de {rolling_window} dias"
            elif demand_view == "Acumulada":
                demand_view_df["valor"] = demand_view_df.groupby("serie")["valor"].cumsum()
                demand_title = "Demanda acumulada del periodo"

            demand_chart = px.line(
                demand_view_df,
                x="periodo",
                y="valor",
                color="serie",
                markers=True,
                title=demand_title,
                labels={"valor": "Demanda"},
            )
            demand_chart.update_xaxes(rangeslider_visible=True)
            market_left.plotly_chart(demand_chart, width="stretch")
        elif current_demand_df.empty:
            market_left.info("No hay datos de demanda disponibles para el periodo seleccionado.")
        else:
            market_left.info("Selecciona al menos una serie de demanda.")

        scatter_df = build_demand_price_scatter(current_demand_df, current_price_df)
        indexed_market_df = build_market_indexed_dataframe(current_demand_df, current_price_df)

        if market_analysis_mode == "Precio diario":
            if not current_price_df.empty:
                price_chart = px.line(
                    current_price_df,
                    x="periodo",
                    y="valor",
                    color="serie",
                    markers=True,
                    title="Precio ponderado de bolsa",
                    labels={"valor": "COP/kWh"},
                )
                price_chart.update_xaxes(rangeslider_visible=True)
                market_right.plotly_chart(price_chart, width="stretch")
            else:
                market_right.info("No hay datos de precio para el periodo seleccionado.")
        elif market_analysis_mode == "Series indexadas":
            if not indexed_market_df.empty:
                indexed_chart = px.line(
                    indexed_market_df,
                    x="periodo",
                    y="indice",
                    color="indicador",
                    markers=True,
                    title="Demanda y precio indexados (base 100)",
                    labels={"indice": "Indice base 100", "periodo": "Fecha"},
                )
                indexed_chart.update_xaxes(rangeslider_visible=True)
                market_right.plotly_chart(indexed_chart, width="stretch")
            else:
                market_right.info("No pude indexar demanda y precio con los datos actuales.")
        else:
            if not scatter_df.empty:
                demand_price_corr = scatter_df["demanda"].corr(scatter_df["precio_bolsa"])
                market_right.metric(
                    "Correlacion demanda-precio",
                    _format_number(demand_price_corr, decimals=2),
                )
                scatter = px.scatter(
                    scatter_df,
                    x="demanda",
                    y="precio_bolsa",
                    color="periodo",
                    title="Relacion entre demanda y precio de bolsa",
                    labels={"demanda": "Demanda", "precio_bolsa": "Precio bolsa"},
                    hover_data={"periodo": True},
                )
                market_right.plotly_chart(scatter, width="stretch")
            else:
                market_right.info("No hay suficientes datos para la dispersion demanda-precio.")

        st.markdown("### Patrones horarios")
        pattern_left, pattern_right = st.columns(2)
        if not heatmap_df.empty:
            heatmap_pivot = (
                heatmap_df.pivot_table(
                    index="dia_semana",
                    columns="hora",
                    values="demanda_promedio",
                    aggfunc="mean",
                )
                .reindex([DAY_LABELS[index] for index in range(1, 8)])
                .sort_index(axis=1)
            )
            heatmap = px.imshow(
                heatmap_pivot,
                aspect="auto",
                color_continuous_scale="YlOrRd",
                labels={"x": "Hora del dia", "y": "Dia de la semana", "color": "Demanda promedio"},
                title="Heatmap horario de demanda",
                text_auto=".0f",
            )
            pattern_left.plotly_chart(heatmap, width="stretch")
        else:
            pattern_left.info("No hay suficientes datos horarios para construir el mapa de calor.")

        if not current_duck_df.empty:
            duck_chart = px.line(
                current_duck_df,
                x="hora",
                y=["demanda_promedio", "demanda_minima", "demanda_maxima"],
                markers=True,
                title="Perfil horario de demanda tipo pato",
                labels={
                    "hora": "Hora del dia",
                    "value": "Demanda",
                    "variable": "Serie",
                },
            )
            duck_chart.update_layout(xaxis=dict(dtick=1))
            pattern_right.plotly_chart(duck_chart, width="stretch")
        else:
            pattern_right.info("No hay datos suficientes para la curva horaria tipo pato.")

        st.caption(
            "La curva de pato se aproxima con la demanda real promedio por hora del dia. "
            "Sirve para destacar el valle de mediodia y el repunte de la tarde aunque no incluya demanda neta solar."
        )

    with hydrology_tab:
        st.subheader("Agua y embalses")
        st.caption("Explora regiones y embalses con enfoque comparativo, sin recurrir a tortas.")

        hydro_region_options = (
            sorted(current_hydrology_df["region"].dropna().unique().tolist())
            if not current_hydrology_df.empty
            else []
        )
        reservoir_options = (
            sorted(current_reservoir_df["embalse"].dropna().unique().tolist())
            if not current_reservoir_df.empty
            else []
        )

        hydro_controls = st.columns(3)
        hydro_focus_regions = hydro_controls[0].multiselect(
            "Regiones a destacar",
            options=hydro_region_options,
            default=hydro_region_options[: min(3, len(hydro_region_options))],
            key="hydro_focus_regions",
        )
        show_historical_average = hydro_controls[1].checkbox(
            "Comparar con media historica",
            value=True,
            key="show_historical_average",
        )
        reservoir_ranking_mode = hydro_controls[2].selectbox(
            "Priorizar embalses",
            options=["Mas bajos", "Mas altos"],
            key="reservoir_ranking_mode",
        )

        left_column, right_column = st.columns(2)

        if not current_hydrology_df.empty:
            hydrology_view_df = current_hydrology_df.copy()
            if hydro_focus_regions:
                hydrology_view_df = hydrology_view_df[hydrology_view_df["region"].isin(hydro_focus_regions)]

            if (
                show_historical_average
                and "media_historica" in hydrology_view_df.columns
                and hydrology_view_df["media_historica"].notna().any()
            ):
                hydrology_long_df = hydrology_view_df.melt(
                    id_vars=["periodo", "region"],
                    value_vars=["aporte_actual", "media_historica"],
                    var_name="serie",
                    value_name="valor",
                )
                hydrology_long_df["serie"] = hydrology_long_df["serie"].map(
                    {
                        "aporte_actual": "Aporte actual",
                        "media_historica": "Media historica",
                    }
                )
                hydrology_chart = px.line(
                    hydrology_long_df,
                    x="periodo",
                    y="valor",
                    color="region",
                    line_dash="serie",
                    markers=True,
                    title="Aportes hidricos por region vs media historica",
                    labels={"valor": "Aporte", "periodo": "Fecha"},
                )
            else:
                hydrology_chart = px.line(
                    hydrology_view_df,
                    x="periodo",
                    y="aporte_actual",
                    color="region",
                    markers=True,
                    title="Aportes hidricos por region",
                    labels={"aporte_actual": "Aporte", "periodo": "Fecha"},
                )
            hydrology_chart.update_xaxes(rangeslider_visible=True)
            left_column.plotly_chart(hydrology_chart, width="stretch")
        else:
            left_column.info("No hay datos hidrologicos disponibles para los filtros seleccionados.")

        if not current_reservoir_df.empty:
            top_reservoir_count = right_column.slider(
                "Embalses en ranking",
                min_value=5,
                max_value=15,
                value=8,
                key="top_reservoir_count",
            )
            latest_df = current_reservoir_df[
                current_reservoir_df["es_ultimo_corte"].astype(str).str.lower() == "true"
            ].copy()
            if not latest_df.empty:
                sort_ascending = reservoir_ranking_mode == "Mas bajos"
                latest_rank_df = latest_df.sort_values("nivel", ascending=sort_ascending).head(top_reservoir_count)
                latest_chart = px.bar(
                    latest_rank_df.sort_values("nivel", ascending=not sort_ascending),
                    x="nivel",
                    y="embalse",
                    orientation="h",
                    title=f"Ultimo corte de nivel por embalse ({reservoir_ranking_mode.lower()})",
                    color="nivel",
                    color_continuous_scale="RdYlGn",
                    labels={"nivel": "Nivel (%)", "embalse": "Embalse"},
                )
                latest_chart.update_layout(coloraxis_colorbar_title="Nivel")
                right_column.plotly_chart(latest_chart, width="stretch")

                default_reservoirs = latest_rank_df["embalse"].tolist()[: min(5, len(latest_rank_df))]
                reservoir_focus = st.multiselect(
                    "Embalses a seguir en el tiempo",
                    options=reservoir_options,
                    default=default_reservoirs,
                    key="reservoir_focus",
                )
                if reservoir_focus:
                    reservoir_view_df = current_reservoir_df[
                        current_reservoir_df["embalse"].isin(reservoir_focus)
                    ].copy()
                    if not reservoir_view_df.empty:
                        reservoir_chart = px.line(
                            reservoir_view_df,
                            x="periodo",
                            y="nivel",
                            color="embalse",
                            markers=True,
                            title="Evolucion de embalses seleccionados",
                            labels={"nivel": "Nivel (%)", "periodo": "Fecha"},
                        )
                        reservoir_chart.update_xaxes(rangeslider_visible=True)
                        st.plotly_chart(reservoir_chart, width="stretch")
                else:
                    st.info("Selecciona al menos un embalse para ver su evolucion temporal.")
            else:
                right_column.info("No se pudo identificar el ultimo corte de embalses.")
        else:
            right_column.info("No hay datos de embalses disponibles para los filtros seleccionados.")

    with schema_tab:
        st.subheader("Explorador del catalogo silver")
        schema_rows = []
        for table_schema in schemas.values():
            for column in table_schema.columns:
                schema_rows.append(
                    {
                        "tabla": table_schema.table_name,
                        "columna": column.name,
                        "tipo": column.data_type,
                    }
                )
        if schema_rows:
            st.dataframe(pd.DataFrame(schema_rows), width="stretch")
        else:
            st.info("No se detectaron columnas en las tablas objetivo.")


if __name__ == "__main__":
    main()
