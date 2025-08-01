# Dockerfile
FROM public.ecr.aws/lambda/python:3.10.5

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Copy model (if including in image, otherwise download from S3)
# If your model is large, stick to S3 download at runtime
COPY model/biomass_model.joblib ${LAMBDA_TASK_ROOT}/model/biomass_model.joblib

# Install the specified packages
COPY requirements.txt .
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Set the CMD to your handler (optional, but good practice)
CMD [ "lambda_function.lambda_handler" ]