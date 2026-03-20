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

```sh
aws sso login --profile ts_terrasacha_admin_access
```

## Docker file

## Create repository

```sh
## test:aws s3 ls --region us-east-1 --profile ts_terrasacha_admin_access
aws ecr create-repository --repository-name ts_biomass_ndvi_lambda_repo --region us-east-1 --profile ts_terrasacha_admin_access
```

- Result

```json
{
    "repository": {
        "repositoryArn": "arn:aws:ecr:us-east-1:879381245127:repository/ts_biomass_ndvi_lambda_repo",
        "registryId": "879381245127",
        "repositoryName": "ts_biomass_ndvi_lambda_repo",
        "repositoryUri": "879381245127.dkr.ecr.us-east-1.amazonaws.com/ts_biomass_ndvi_lambda_repo",
        "createdAt": "2025-12-21T19:25:09.653000-05:00",
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
aws ecr get-login-password --region us-east-1 --profile ts_terrasacha_admin_access | docker login --username AWS --password-stdin 879381245127.dkr.ecr.us-east-1.amazonaws.com 

docker build -t ts_biomass_ndvi_lambda_image .

docker tag ts_biomass_ndvi_lambda_image:latest 879381245127.dkr.ecr.us-east-1.amazonaws.com/ts_biomass_ndvi_lambda_repo:latest

docker push 879381245127.dkr.ecr.us-east-1.amazonaws.com/ts_biomass_ndvi_lambda_repo:latest --profile ts_terrasacha_admin_access
```

## Create lambda function

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

> result

```json
{
    "FunctionName": "ts_biomass_ndvi_lambda",
    "FunctionArn": "arn:aws:lambda:us-east-1:879381245127:function:ts_biomass_ndvi_lambda",
    "Role": "arn:aws:iam::879381245127:role/ts-lambda-models-execution-role",
    "CodeSize": 0,
    "Description": "",
    "Timeout": 30,
    "MemorySize": 512,
    "LastModified": "2025-12-22T13:54:55.451+0000",
    "CodeSha256": "24218cee690f31fff86a8d9b123a330093b233e79a40ee97c65635a3369fb355",
    "Version": "$LATEST",
    "TracingConfig": {
        "Mode": "PassThrough"
    },
    "RevisionId": "2533a7e2-6b5c-4321-9b7f-fb309d7737b3",
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

### Update Lambda Function Code

```sh
aws lambda update-function-code \
    --function-name ts_biomass_ndvi_lambda \
    --image-uri 879381245127.dkr.ecr.us-east-1.amazonaws.com/ts_biomass_ndvi_lambda_repo:latest
```

### Update Lambda Function Configuration (Timeout and Memory)

To update the timeout and memory-size of an existing Lambda function, use the `update-function-configuration` command:

```sh
# Update timeout (in seconds, max 900 seconds / 15 minutes)
aws lambda update-function-configuration \
    --function-name ts_biomass_ndvi_lambda \
    --timeout 60

# Update memory-size (in MB, between 128 and 10240, must be a multiple of 1MB)
aws lambda update-function-configuration \
    --function-name ts_biomass_ndvi_lambda \
    --memory-size 1024

# Update both timeout and memory-size in a single command
aws lambda update-function-configuration \
    --function-name ts_biomass_ndvi_lambda \
    --timeout 120 \
    --memory-size 1024 \
    --profile suan-blockchain

aws lambda update-function-configuration \
    --function-name ts_biomass_ndvi_lambda \
    --timeout 120 \
    --memory-size 2048 \
    --profile suan-blockchain
```

**Notes:**
- **Timeout**: Range is 1-900 seconds (15 minutes). Default is 3 seconds.
- **Memory-size**: Range is 128-10240 MB, must be a multiple of 1MB. Default is 128 MB.
- More memory also increases CPU power proportionally, which can improve performance.

## Test Lambda Function

There are several ways to test your Lambda function:

### Method 1: Direct Lambda Invocation (AWS CLI)

Test the function directly using the AWS CLI with a test event file:

```sh
# Invoke the function with a test event
aws lambda invoke \
    --function-name ts_biomass_ndvi_lambda \
    --cli-binary-format raw-in-base64-out \
    --payload file://EventTest.json \
    --region us-east-1 \
    --profile ts_terrasacha_admin_access \
    response.json

# View the response
cat response.json
```

**EventTest.json structure:**
```json
{
  "Records": [
    {
      "s3": {
        "bucket": {
          "name": "tsbiomassmodeldata"
        },
        "object": {
          "key": "your_test_file.tif"
        }
      }
    }
  ],
  "custom_payload": "{\"output_bucket_name\": \"tsbiomassmodeldata\", \"model_bucket_name\": \"tsbiomassmodeldata\"}"
}
```

**Note:** Replace `your_test_file.tif` with an actual file that exists in your S3 bucket.

### Method 2: Test with Inline Payload

You can also pass the event payload directly:

```sh
aws lambda invoke \
    --function-name ts_biomass_ndvi_lambda_no_png \
    --payload '{
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
    --region us-east-1 \
    response.json
  
  aws lambda invoke \
    --function-name ts_biomass_ndvi_lambda_no_png \
    --payload '{
      "Records": [
        {
          "s3": {
            "bucket": {
              "name": "tsbiomassmodeldata"
            },
            "object": {
              "key": "img__20251120160455__S2__B4_B3_B2__2025_09_09__9155.tif"
            }
          }
        }
      ],
      "custom_payload": "{\"output_bucket_name\": \"tsbiomassmodeldata\", \"model_bucket_name\": \"tsbiomassmodeldata\"}"
    }' \
    --region us-east-1 \
    response.json
```

### Method 3: View Lambda Logs

Monitor the function execution and debug issues:

```sh
# View recent logs
aws logs tail /aws/lambda/ts_biomass_ndvi_lambda_no_png --follow --region us-east-1

# View logs for a specific time period
aws logs tail /aws/lambda/ts_biomass_ndvi_lambda \
    --since 1h \
    --region us-east-1
    --profile ts_terrasacha_admin_access

# Get the last 50 log entries
aws logs tail /aws/lambda/ts_biomass_ndvi_lambda \
    --since 1h \
    --format short \
    --region us-east-1 | tail -50
```

### Method 4: Test via API Gateway

If your Lambda is connected to API Gateway, you can test it via HTTP:

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
            "key": "your_test_file.tif"
          }
        }
      }
    ],
    "custom_payload": "{\"output_bucket_name\": \"tsbiomassmodeldata\", \"model_bucket_name\": \"tsbiomassmodeldata\"}"
  }' \
  https://9e7wnzvwcb.execute-api.us-east-1.amazonaws.com/dev/predict_nvdi_tif
```

### Verify Output

After testing, verify that the output file was created in S3:

```sh
# List files in the bucket to find the output
aws s3 ls s3://tsbiomassmodeldata/ | grep biomass_map

# Download the output file
aws s3 cp s3://tsbiomassmodeldata/biomass_map_your_test_file.tif ~/Downloads/
```

### Check Function Status

Before testing, ensure the function is active:

```sh
aws lambda get-function \
    --function-name ts_biomass_ndvi_lambda \
    --region us-east-1
```

**Expected Response:**
- `State`: Should be `Active`
- `LastUpdateStatus`: Should be `Successful`

```json
{
    "FunctionName": "ts_biomass_ndvi_lambda",
    "FunctionArn": "arn:aws:lambda:us-east-1:879381245127:function:ts_biomass_ndvi_lambda",
    "Role": "arn:aws:iam::879381245127:role/ts-lambda-models-execution-role",
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
    --code ImageUri=879381245127.dkr.ecr.us-east-1.amazonaws.com/ts_biomass_ndvi_lambda_repo@sha256:0b918a99e3911e9f462c36c34f4ca3cbbd7e88baed53e5aa2bbd3e30c1247e41 \
    --role arn:aws:iam::879381245127:role/ts-lambda-models-execution-role \
    --timeout 30 \
    --memory-size 512

aws lambda update-function-configuration \
  --function-name ts_biomass_ndvi_lambda_no_png \
  --timeout 120 \
  --memory-size 2048

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

### Pruebas polígonos grandes @VH

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
            "key": "img__20251120160455__S2__B4_B3_B2__2025_09_09__9155.tif"
          }
        }
      }
    ],
    "custom_payload": "{\"output_bucket_name\": \"tsbiomassmodeldata\", \"model_bucket_name\": \"tsbiomassmodeldata\"}"
}' \
  https://9e7wnzvwcb.execute-api.us-east-1.amazonaws.com/dev/predict_nvdi_tif


```

### Generate png and download it

```bash

# Check if the biomass image was generated

aws s3 ls s3://tsbiomassmodeldata/ | grep img__20251120160455__S2__B4_B3_B2__2025_09_09__9155

biomass_map_img__20251120160455__S2__B4_B3_B2__2025_09_09__9155.tif

curl -X POST https://9e7wnzvwcb.execute-api.us-east-1.amazonaws.com/dev/util_export_png \
  -H "Content-Type: application/json" \
  -d '{
    "bucket": "tsbiomassmodeldata",
    "key": "biomass_map_img__20251120160455__S2__B4_B3_B2__2025_09_09__9155.tif"
  }'

# Result

{"statusCode": 200, "body": "{\"message\": \"Successfully converted biomass_map_img__20251120160455__S2__B4_B3_B2__2025_09_09__9155.tif to PNG\", \"input_location\": \"s3://tsbiomassmodeldata/biomass_map_img__20251120160455__S2__B4_B3_B2__2025_09_09__9155.tif\", \"output_location\": \"s3://tsbiomassmodeldata/png_biomass_map_img__20251120160455__S2__B4_B3_B2__2025_09_09__9155.png\"}"}% 

# Download the .png

aws s3 cp s3://tsbiomassmodeldata/png_biomass_map_img__20251120160455__S2__B4_B3_B2__2025_09_09__9155.png ~/Downloads/ --profile suan-blockchain

```


## Testing

```sh
aws lambda invoke \
  --function-name ts_biomass_ndvi_lambda \
  --cli-binary-format raw-in-base64-out \
  --payload file://EventTest.json \
  --region us-east-1 \
  --profile 879381245127_AdministratorAccess \
  response.json

```