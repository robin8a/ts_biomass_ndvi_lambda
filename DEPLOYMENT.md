# Deploy

```sh
# Install 
brew install pyenv-virtualenv

# Create a virtual environment
# pyenv virtualenv <python_version> <environment_name>
pyenv virtualenv 3.10.5 ts_biomass_ndvi_lambda
pyenv activate ts_biomass_ndvi_lambda

pyenv versions

# VSC
cmd + shift + P 
select interpreter
<ts_biomass_ndvi_lambda>

```

## Install libraries to create the model

```sh
# pip install numpy pandas sklearn joblib 
pip install rasterio numpy pandas scikit-learn joblib boto3 matplot

```

## Save and create model

```sh
python biomass_model_script.py
```

## Create a requirements.txt file

```requirements.txt
rasterio
boto3
scikit-learn
numpy
pandas
joblib
```

## Create Your Lambda Handler File (lambda_function.py)

## Install Dependencies into a package Directory

```sh
mkdir package
pip install -r requirements.txt --target package/
```

## Create the Deployment Zip File

```sh
# Go into the package directory first, zip its contents
cd package
zip -r ../biomass_model_deployment_package.zip .

# Go back to the root, add your lambda_function.py and model directory
cd ..
zip -g biomass_model_deployment_package.zip lambda_function.py
zip -r -g biomass_model_deployment_package.zip model/

```

## Docker file

## Create repository

```sh
aws ecr create-repository --repository-name ts_biomass_ndvi_lambda_repo --region us-east-1
```

- Result

```json
{
    "repository": {
        "repositoryArn": "arn:aws:ecr:us-east-1:036134507423:repository/ts_biomass_ndvi_lambda_repo",
        "registryId": "036134507423",
        "repositoryName": "ts_biomass_ndvi_lambda_repo",
        "repositoryUri": "036134507423.dkr.ecr.us-east-1.amazonaws.com/ts_biomass_ndvi_lambda_repo",
        "createdAt": "2025-08-01T17:21:00.982000-05:00",
        "imageTagMutability": "MUTABLE",
        "imageScanningConfiguration": {
            "scanOnPush": false
        },
        "encryptionConfiguration": {
            "encryptionType": "AES256"
        }
    }
}

```

## Docker build and AWS ECR push

```sh
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 036134507423.dkr.ecr.us-east-1.amazonaws.com

docker build -t ts_biomass_ndvi_lambda_image .

docker tag ts_biomass_ndvi_lambda_image:latest 036134507423.dkr.ecr.us-east-1.amazonaws.com/ts_biomass_ndvi_lambda_repo:latest

docker push 036134507423.dkr.ecr.us-east-1.amazonaws.com/ts_biomass_ndvi_lambda_repo:latest
```

## Create lambda function

```sh
aws lambda create-function \
    --function-name ts_biomass_ndvi_lambda \
    --package-type Image \
    --code ImageUri=036134507423.dkr.ecr.us-east-1.amazonaws.com/ts_biomass_ndvi_lambda_repo:latest \
    --role arn:aws:iam::036134507423:role/ts-lambda-biomass-execution-role \
    --timeout 30 \
    --memory-size 512

```

> result

```json
{
    "FunctionName": "ts_biomass_ndvi_lambda",
    "FunctionArn": "arn:aws:lambda:us-east-1:036134507423:function:ts_biomass_ndvi_lambda",
    "Role": "arn:aws:iam::036134507423:role/ts-lambda-biomass-execution-role",
    "CodeSize": 0,
    "Description": "",
    "Timeout": 30,
    "MemorySize": 512,
    "LastModified": "2025-08-02T03:47:56.892+0000",
    "CodeSha256": "0b918a99e3911e9f462c36c34f4ca3cbbd7e88baed53e5aa2bbd3e30c1247e41",
    "Version": "$LATEST",
    "TracingConfig": {
        "Mode": "PassThrough"
    },
    "RevisionId": "e002bd58-3b9c-40b9-a399-808518425dd9",
    "State": "Pending",
    "StateReason": "The function is being created.",
    "StateReasonCode": "Creating",
    "PackageType": "Image",
    "Architectures": [
        "x86_64"
    ],
    "EphemeralStorage": {
        "Size": 512
    },
    "SnapStart": {
        "ApplyOn": "None",
        "OptimizationStatus": "Off"
    }
}
```

## Update lambda function

```sh
aws lambda update-function-code \
    --package-type Image \
    --function-name ts_biomass_ndvi_lambda

aws lambda create-function \
    --function-name ts_biomass_ndvi_lambda \
    --package-type Image \
    --code ImageUri=036134507423.dkr.ecr.us-east-1.amazonaws.com/ts_biomass_ndvi_lambda_repo:latest \
    --role arn:aws:iam::036134507423:role/ts-lambda-biomass-execution-role \
    --timeout 30 \
    --memory-size 512


aws lambda update-function-code \
    --function-name ts_biomass_ndvi_lambda \
    --image-uri 036134507423.dkr.ecr.us-east-1.amazonaws.com/ts_biomass_ndvi_lambda_repo:latest
```

```json
{
    "FunctionName": "ts_biomass_ndvi_lambda",
    "FunctionArn": "arn:aws:lambda:us-east-1:036134507423:function:ts_biomass_ndvi_lambda",
    "Role": "arn:aws:iam::036134507423:role/ts-lambda-biomass-execution-role",
    "CodeSize": 0,
    "Description": "",
    "Timeout": 30,
    "MemorySize": 512,
    "LastModified": "2025-08-04T21:28:26.000+0000",
    "CodeSha256": "24218cee690f31fff86a8d9b123a330093b233e79a40ee97c65635a3369fb355",
    "Version": "$LATEST",
    "TracingConfig": {
        "Mode": "PassThrough"
    },
    "RevisionId": "26e06c06-c2d6-4373-908a-f39121257a9f",
    "State": "Active",
    "LastUpdateStatus": "InProgress",
    "LastUpdateStatusReason": "The function is being created.",
    "LastUpdateStatusReasonCode": "Creating",
    "PackageType": "Image",
    "Architectures": [
        "x86_64"
    ],
    "EphemeralStorage": {
        "Size": 512
    },
    "SnapStart": {
        "ApplyOn": "None",
        "OptimizationStatus": "Off"
    }
}
```

```sh
aws lambda create-function \
    --function-name ts_biomass_ndvi_lambda_no_png \
    --package-type Image \
    --code ImageUri=036134507423.dkr.ecr.us-east-1.amazonaws.com/ts_biomass_ndvi_lambda_repo@sha256:0b918a99e3911e9f462c36c34f4ca3cbbd7e88baed53e5aa2bbd3e30c1247e41 \
    --role arn:aws:iam::036134507423:role/ts-lambda-biomass-execution-role \
    --timeout 30 \
    --memory-size 512

```

```sh
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"features": [0.7, 0.5]}' \
  https://9e7wnzvwcb.execute-api.us-east-1.amazonaws.com/dev/predict_nvdi_tif



curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "Records": [
      {
        "s3": {
          "bucket": {
            "name": "tsbiomassmodeldata"
          },
          "object": {
            "key": "pol_20250215180043_2022_S2_B2_B3_B4_B5_drive.tif"
          }
        }
      }
    ],
    "custom_payload": "{\"output_bucket_name\": \"tsbiomassmodeldata\", \"model_bucket_name\": \"tsbiomassmodeldata\"}"
}' \
  https://9e7wnzvwcb.execute-api.us-east-1.amazonaws.com/dev/predict_nvdi_tif}' \
  https://9e7wnzvwcb.execute-api.us-east-1.amazonaws.com/dev/predict_nvdi_tif



curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "Records": [
      {
        "s3": {
          "bucket": {
            "name": "tsbiomassmodeldata"
          },
          "object": {
            "key": "pol_20250215003502_2024_S2_B2_B3_B4_drive.tif"
          }
        }
      }
    ],
    "custom_payload": "{\"output_bucket_name\": \"tsbiomassmodeldata\", \"model_bucket_name\": \"tsbiomassmodeldata\"}"
}' \
  https://9e7wnzvwcb.execute-api.us-east-1.amazonaws.com/dev/predict_nvdi_tif


curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "Records": [
      {
        "s3": {
          "bucket": {
            "name": "tsbiomassmodeldata"
          },
          "object": {
            "key": "img__20251016212350__S2__B4_B3_B2__2023_01_28__2336.tif"
          }
        }
      }
    ],
    "custom_payload": "{\"output_bucket_name\": \"tsbiomassmodeldata\", \"model_bucket_name\": \"tsbiomassmodeldata\"}"
}' \
  https://9e7wnzvwcb.execute-api.us-east-1.amazonaws.com/dev/predict_nvdi_tif


curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "Records": [
      {
        "s3": {
          "bucket": {
            "name": "tsbiomassmodeldata"
          },
          "object": {
            "key": "img__20251105162753__S2__B4_B3_B2__2025_10_01__5226.tif"
          }
        }
      }
    ],
    "custom_payload": "{\"output_bucket_name\": \"tsbiomassmodeldata\", \"model_bucket_name\": \"tsbiomassmodeldata\"}"
}' \
  https://9e7wnzvwcb.execute-api.us-east-1.amazonaws.com/dev/predict_nvdi_tif

# File generated: 

biomass_map_img__20251016212350__S2__B4_B3_B2__2023_01_28__2336.tif

aws s3 cp s3://tsbiomassmodeldata/super_resolution/super_resolution_2f980717_S2_07.tif ~/Downloads/ --profile suan-blockchain

img__20251105162753__S2__B4_B3_B2__2025_10_01__5226.tif

aws s3 ls s3://tsbiomassmodeldata/ | grep img__20251105162753__S2__B4_B3_B2__2025_10_01__5226

aws s3 cp s3://tsbiomassmodeldata/biomass_map_img__20251105162753__S2__B4_B3_B2__2025_10_01__5226.tif ~/Downloads/ --profile suan-blockchain

aws s3 ls s3://tsbiomassmodeldata/ | grep pol_20250215003502_2024_S2_B2_B3_B4

aws s3 cp s3://tsbiomassmodeldata/biomass_map_pol_20250215003502_2024_S2_B2_B3_B4_drive.tif ~/Downloads/ --profile suan-blockchain


biomass_map_pol_20250215003502_2024_S2_B2_B3_B4_drive.tif

curl -X POST https://9e7wnzvwcb.execute-api.us-east-1.amazonaws.com/dev/util_export_png \
  -H "Content-Type: application/json" \
  -d '{
    "bucket": "tsbiomassmodeldata",
    "key": "biomass_map_pol_20250215003502_2024_S2_B2_B3_B4_drive.tif"
  }'

{"statusCode": 200, "body": "{\"message\": \"Successfully converted biomass_map_pol_20250215003502_2024_S2_B2_B3_B4_drive.tif to PNG\", \"input_location\": \"s3://tsbiomassmodeldata/biomass_map_pol_20250215003502_2024_S2_B2_B3_B4_drive.tif\", \"output_location\": \"s3://tsbiomassmodeldata/png_biomass_map_pol_20250215003502_2024_S2_B2_B3_B4_drive.png\"}"}

aws s3 cp s3://tsbiomassmodeldata/png_biomass_map_pol_20250215003502_2024_S2_B2_B3_B4_drive.png ~/Downloads/ --profile suan-blockchain

# It works

```

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "Records": [
      {
        "s3": {
          "bucket": {
            "name": "tsbiomassmodeldata"
          },
          "object": {
            "key": "img__20251105162753__S2__B4_B3_B2__2025_10_01__5226.tif"
          }
        }
      }
    ],
    "custom_payload": "{\"output_bucket_name\": \"tsbiomassmodeldata\", \"model_bucket_name\": \"tsbiomassmodeldata\"}"
}' \
  https://9e7wnzvwcb.execute-api.us-east-1.amazonaws.com/dev/predict_nvdi_tif

aws s3 ls s3://tsbiomassmodeldata/ | grep img__20251105162753__S2__B4_B3_B2__2025_10_01__5226

aws s3 cp s3://tsbiomassmodeldata/biomass_map_img__20251105162753__S2__B4_B3_B2__2025_10_01__5226.tif ~/Downloads/ --profile suan-blockchain

curl -X POST https://9e7wnzvwcb.execute-api.us-east-1.amazonaws.com/dev/util_export_png \
  -H "Content-Type: application/json" \
  -d '{
    "bucket": "tsbiomassmodeldata",
    "key": "biomass_map_img__20251105162753__S2__B4_B3_B2__2025_10_01__5226.tif"
  }'

{"statusCode": 200, "body": "{\"message\": \"Successfully converted biomass_map_img__20251105162753__S2__B4_B3_B2__2025_10_01__5226.tif to PNG\", \"input_location\": \"s3://tsbiomassmodeldata/biomass_map_img__20251105162753__S2__B4_B3_B2__2025_10_01__5226.tif\", \"output_location\": \"s3://tsbiomassmodeldata/png_biomass_map_img__20251105162753__S2__B4_B3_B2__2025_10_01__5226.png\"}"}

aws s3 cp s3://tsbiomassmodeldata/png_biomass_map_img__20251105162753__S2__B4_B3_B2__2025_10_01__5226.png ~/Downloads/ --profile suan-blockchain


```

```bash

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "Records": [
      {
        "s3": {
          "bucket": {
            "name": "tsbiomassmodeldata"
          },
          "object": {
            "key": "pol_20250215003502_2024_S2_B2_B3_B4_drive.tif"
          }
        }
      }
    ],
    "custom_payload": "{\"output_bucket_name\": \"tsbiomassmodeldata\", \"model_bucket_name\": \"tsbiomassmodeldata\"}"
}' \
  https://9e7wnzvwcb.execute-api.us-east-1.amazonaws.com/dev/predict_nvdi_tif

aws s3 ls s3://tsbiomassmodeldata/ | grep pol_20250215003502_2024_S2_B2_B3_B4_drive

```

```sh

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "Records": [
      {
        "s3": {
          "bucket": {
            "name": "tsbiomassmodeldata"
          },
          "object": {
            "key": "img__20251110191822__S2__B4_B3_B2__2025_10_06__2827.tif"
          }
        }
      }
    ],
    "custom_payload": "{\"output_bucket_name\": \"tsbiomassmodeldata\", \"model_bucket_name\": \"tsbiomassmodeldata\"}"
}' \
  https://9e7wnzvwcb.execute-api.us-east-1.amazonaws.com/dev/predict_nvdi_tif


aws s3 ls s3://tsbiomassmodeldata/ | grep img__20251110191822__S2__B4_B3_B2__2025_10_06__2827

curl -X POST https://9e7wnzvwcb.execute-api.us-east-1.amazonaws.com/dev/util_export_png \
  -H "Content-Type: application/json" \
  -d '{
    "bucket": "tsbiomassmodeldata",
    "key": "biomass_map_img__20251110191822__S2__B4_B3_B2__2025_10_06__2827.tif"
  }'


aws s3 cp s3://tsbiomassmodeldata/png_biomass_map_img__20251110191822__S2__B4_B3_B2__2025_10_06__2827.png ~/Downloads/ --profile suan-blockchain


```