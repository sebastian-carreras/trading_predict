FROM apache/airflow:2.8.1-python3.10

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         git \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow

# Copiar requirements de airflow
COPY dockerfiles/airflow/requirements.txt /requirements.txt
RUN pip install --user --upgrade pip
RUN pip install --no-cache-dir --user apache-airflow[virtualenv]==2.8.1 -r /requirements.txt

# Copiar código fuente del proyecto trading_predict
COPY --chown=airflow:root src /opt/airflow/src
COPY --chown=airflow:root requirements.txt /opt/airflow/requirements_project.txt

# Instalar dependencias del proyecto
RUN pip install --no-cache-dir --user -r /opt/airflow/requirements_project.txt
