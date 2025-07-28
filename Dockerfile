# ----------------------------------------
# Base image with Python 3.10
FROM --platform=linux/amd64 python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and folders
COPY app/ ./app/
COPY pdfs/ ./pdfs/
COPY csv/ ./csv/
COPY output/ ./output/

# Copy model files into the container
COPY enhanced_pdf_heading_rf_model.joblib .
COPY enhanced_label_encoder.joblib .
COPY enhanced_model_metadata.json .

# Default command to run the pipeline
CMD ["python", "app/pipeline.py"]
