# SIMEM Streamlit Dashboards

Proyecto separado para explorar tablas `silver` y `gold` de SIMEM con `Streamlit`, `Athena` y filtros interactivos.

## Que incluye

- Dashboard ejecutivo con series de demanda y precio.
- Dashboard tematico de El Nino 2023-2024 con dependencia termica y costo estimado en CO2.
- Dashboard de oferta con barras comparativas y concentracion acumulada por planta.
- Dashboard de mercado con vistas diaria, acumulada y promedio movil, ademas de relacion demanda-precio.
- Dashboard hidrologico y de embalses.
- Filtros laterales por fecha, tipo de generacion, estado del recurso y region.
- Explorador de schema para revisar que columnas encontro Athena en `simem_silver`.

## Estructura

- [app.py](C:/proyecto_final_maestria/simem_streamlit_dashboards/app.py): app principal de Streamlit.
- [simem_dashboard/athena.py](C:/proyecto_final_maestria/simem_streamlit_dashboards/simem_dashboard/athena.py): cliente liviano para consultas en Athena.
- [simem_dashboard/schema.py](C:/proyecto_final_maestria/simem_streamlit_dashboards/simem_dashboard/schema.py): descubrimiento de schema y utilidades de columnas.
- [simem_dashboard/settings.py](C:/proyecto_final_maestria/simem_streamlit_dashboards/simem_dashboard/settings.py): configuracion via variables de entorno.

## Requisitos

- Python 3.11 o superior
- Credenciales AWS disponibles en la sesion
- Permisos para `Athena`, `Glue Data Catalog` y lectura del bucket de resultados

## Variables de entorno

Opcionales:

- `AWS_REGION=us-east-1`
- `ATHENA_WORKGROUP=primary`
- `ATHENA_OUTPUT=s3://eafit-proyecto-integrador-simem/athena-results/`
- `ATHENA_DATABASE=simem_silver`
- `ATHENA_GOLD_DATABASE=simem_gold`
- `ATHENA_POLL_INTERVAL=1.0`

Ejemplo en PowerShell:

```powershell
$env:AWS_REGION = "us-east-1"
$env:ATHENA_WORKGROUP = "primary"
$env:ATHENA_OUTPUT = "s3://eafit-proyecto-integrador-simem/athena-results/"
$env:ATHENA_DATABASE = "simem_silver"
$env:ATHENA_GOLD_DATABASE = "simem_gold"
```

## Instalacion

```powershell
cd C:\proyecto_final_maestria\simem_streamlit_dashboards
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Ejecucion

```powershell
cd C:\proyecto_final_maestria\simem_streamlit_dashboards
.venv\Scripts\Activate.ps1
streamlit run app.py
```

## Preguntas de negocio que responde

- Como se compone la matriz operativa por tipo de generacion.
- Como evoluciona la demanda real frente a la demanda comercial.
- Como se comporta el precio ponderado de bolsa en el tiempo.
- Que regiones hidrologicas o embalses muestran mas tension reciente.
- En que momento El Nino 2023-2024 disparo las emisiones del SIN y cuanto CO2 adicional implico la dependencia termica.

## Notas

- La app inspecciona `information_schema.columns` para adaptarse a las columnas reales que existan en Athena.
- Si una tabla no tiene las columnas esperadas, la app no falla: muestra una advertencia con lo que encontro.
