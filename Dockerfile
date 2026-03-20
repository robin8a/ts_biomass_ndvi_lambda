# Dockerfile
FROM public.ecr.aws/lambda/python:3.10

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Copy model (if including in image, otherwise download from S3)
# If your model is large, stick to S3 download at runtime
COPY model/biomass_model.joblib ${LAMBDA_TASK_ROOT}/model/biomass_model.joblib

# Install the specified packages
COPY requirements.txt .
# Install deps into the Lambda task root.
# NOTE: Some dependency graphs may (incorrectly) pull in the legacy `typing` backport.
# If it lands as `${LAMBDA_TASK_ROOT}/typing.py`, it will shadow the stdlib `typing`
# and can crash at import-time (e.g. `_abc_registry` missing).
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}" \
  && rm -f "${LAMBDA_TASK_ROOT}/typing.py" "${LAMBDA_TASK_ROOT}/typing.pyi" \
  && rm -rf "${LAMBDA_TASK_ROOT}/typing" \
  && rm -rf "${LAMBDA_TASK_ROOT}"/typing-*.dist-info "${LAMBDA_TASK_ROOT}"/typing-*.egg-info || true

# Set the CMD to your handler (optional, but good practice)
CMD [ "lambda_function.lambda_handler" ]