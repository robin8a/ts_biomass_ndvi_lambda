# SIMPLE_DEPLOY.md

Minimal deploy/update steps for AWS Lambda function `ts_biomass_ndvi_lambda` (container image).

## Assumptions
- AWS region: `us-east-1`
- AWS profile: `879381245127_AdministratorAccess`
- ECR repository: `ts_biomass_ndvi_lambda_repo`
- Lambda image role: `arn:aws:iam::879381245127:role/ts-lambda-models-execution-role`

## 0) Login (SSO)
```sh
aws sso login --profile 879381245127_AdministratorAccess
```

## 1) Login Docker to ECR
```sh
aws ecr get-login-password --region us-east-1 --profile 879381245127_AdministratorAccess | \
  docker login --username AWS --password-stdin \
  879381245127.dkr.ecr.us-east-1.amazonaws.com
```

## 2) (Optional) Create the ECR repository
```sh
aws ecr create-repository \
  --repository-name ts_biomass_ndvi_lambda_repo \
  --region us-east-1 \
  --profile 879381245127_AdministratorAccess
```

## 3) Build, tag, and push the container image
From this repo directory:
```sh
docker build -t ts_biomass_ndvi_lambda_image .

docker tag ts_biomass_ndvi_lambda_image:latest \
  879381245127.dkr.ecr.us-east-1.amazonaws.com/ts_biomass_ndvi_lambda_repo:latest

docker push \
  879381245127.dkr.ecr.us-east-1.amazonaws.com/ts_biomass_ndvi_lambda_repo:latest \
  --profile 879381245127_AdministratorAccess
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
  --profile 879381245127_AdministratorAccess
```

## 5) Update the Lambda function code (every deploy)
```sh
aws lambda update-function-code \
  --function-name ts_biomass_ndvi_lambda \
  --image-uri 879381245127.dkr.ecr.us-east-1.amazonaws.com/ts_biomass_ndvi_lambda_repo:latest \
  --profile 879381245127_AdministratorAccess
```

## 6) (Optional) Update Lambda configuration (timeout/memory)
```sh
aws lambda update-function-configuration \
  --function-name ts_biomass_ndvi_lambda \
  --timeout 60 \
  --memory-size 1024 \
  --profile 879381245127_AdministratorAccess
```

## 7) Verify
```sh
aws lambda get-function \
  --function-name ts_biomass_ndvi_lambda \
  --region us-east-1 \
  --profile 879381245127_AdministratorAccess
```

