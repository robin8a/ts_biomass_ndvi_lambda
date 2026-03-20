# SIMPLE_DEPLOY.md

Minimal deploy/update steps for AWS Lambda function `ts_biomass_ndvi_lambda` (container image).

## Assumptions
- AWS region: `us-east-1`
- AWS profile: `ts_terrasacha_admin_access`
- ECR repository: `ts_biomass_ndvi_lambda_repo`
- Lambda image role: `arn:aws:iam::879381245127:role/ts-lambda-models-execution-role`

## 0) Login (SSO)
```sh
aws sso login --profile ts_terrasacha_admin_access
```

## 1) Login Docker to ECR
```sh
aws ecr get-login-password --region us-east-1 --profile ts_terrasacha_admin_access | \
  docker login --username AWS --password-stdin \
  879381245127.dkr.ecr.us-east-1.amazonaws.com
```

## 2) (Optional) Create the ECR repository
```sh
aws ecr create-repository \
  --repository-name ts_biomass_ndvi_lambda_repo \
  --region us-east-1 \
  --profile ts_terrasacha_admin_access
```

## 3) Build, tag, and push the container image
From this repo directory:
```sh
docker build -t ts_biomass_ndvi_lambda_image .

docker tag ts_biomass_ndvi_lambda_image:latest \
  879381245127.dkr.ecr.us-east-1.amazonaws.com/ts_biomass_ndvi_lambda_repo:latest

docker push \
  879381245127.dkr.ecr.us-east-1.amazonaws.com/ts_biomass_ndvi_lambda_repo:latest \
  --profile ts_terrasacha_admin_access
```

## 4) Create the Lambda function (one-time)
Run once if the Lambda does not exist yet:
```sh
aws lambda create-function \
  --function-name ts_biomass_ndvi_lambda \
  --package-type Image \
  --code ImageUri=879381245127.dkr.ecr.us-east-1.amazonaws.com/ts_biomass_ndvi_lambda_repo:latest \
  --role arn:aws:iam::879381245127:role/ts-lambda-models-execution-role \
  --timeout 30 \
  --memory-size 512 \
  --profile ts_terrasacha_admin_access
```

## 5) Update the Lambda function code (every deploy)
```sh
aws lambda update-function-code \
  --function-name ts_biomass_ndvi_lambda \
  --image-uri 879381245127.dkr.ecr.us-east-1.amazonaws.com/ts_biomass_ndvi_lambda_repo:latest \
  --profile ts_terrasacha_admin_access
```

## 6) (Optional) Update Lambda configuration (timeout/memory)
```sh
aws lambda update-function-configuration \
  --function-name ts_biomass_ndvi_lambda \
  --timeout 60 \
  --memory-size 1024 \
  --profile ts_terrasacha_admin_access
```

## 7) Verify
```sh
aws lambda get-function \
  --function-name ts_biomass_ndvi_lambda \
  --region us-east-1 \
  --profile ts_terrasacha_admin_access
```

