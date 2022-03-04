FROM python:3.9-slim-bullseye
LABEL maintainer="Satyam"
ENV SERVER_HOST 0.0.0.0
ENV ARTIFACTS_STORE mlruns/
RUN mkdir -p opt/app
WORKDIR opt/app
COPY . .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && rm housing_value-0.0.0-py3-none-any.whl requirements.txt \
    && apt-get clean
RUN chmod +x /opt/app/run.sh
ENTRYPOINT ["/usr/bin/env"]
CMD ["bash", "/opt/app/run.sh"]