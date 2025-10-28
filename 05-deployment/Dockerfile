# Base image from the Zoomcamp course
FROM agrigorev/zoomcamp-model:2025

# Set working directory
WORKDIR /code

# Copy dependency files first
COPY "pyproject.toml" "uv.lock" ./

# Install uv and dependencies properly
RUN pip install --upgrade pip
RUN pip install scikit-learn==1.6.1 fastapi uvicorn pydantic joblib requests

# Copy the rest
COPY "predict.py" "pipeline_v1.bin" ./

EXPOSE 9696

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]

