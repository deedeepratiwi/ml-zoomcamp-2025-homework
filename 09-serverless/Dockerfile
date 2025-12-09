# Start from the base image provided in the homework description
FROM agrigorev/model-2024-hairstyle:v3

# Set the working directory (optional, but good practice)
WORKDIR /var/task

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --upgrade --no-cache-dir

# Copy the lambda handler code
COPY lambda_function.py .

COPY hair_classifier_empty.onnx.data .
COPY hair_classifier_empty.onnx .

# Set the command for the Lambda function handler
# The format is [filename].[handler_function_name]
CMD ["lambda_function.lambda_handler"]