# The service uses the standard library only, so there is nothing to install.
FROM python:3.12-slim
WORKDIR /app
RUN useradd --system --no-create-home agrosense
COPY agrosense_ia/ agrosense_ia/
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
USER agrosense
EXPOSE 8000
CMD ["python", "-m", "agrosense_ia"]
