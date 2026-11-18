FROM python:3.13-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy onnxruntime fastapi uvicorn
COPY pii ./pii
COPY model/tagger.int8.onnx model/vectors.npz model/first_names.txt ./model/
COPY api.py .
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
