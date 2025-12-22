# Biomass Calculation Lambda Function

A serverless AWS Lambda function that calculates biomass estimates from multi-band GeoTIFF files using NDVI (Normalized Difference Vegetation Index) and EVI (Enhanced Vegetation Index) data through a trained machine learning model.

## Overview

This project provides an automated, scalable solution for processing satellite imagery to estimate biomass. The Lambda function processes multi-band GeoTIFF files stored in S3, extracts NDVI and EVI bands, applies a pre-trained scikit-learn model to predict biomass values for each pixel, and outputs a single-band GeoTIFF containing the biomass estimates.

## How Biomass Calculation Works

### Algorithm Overview

The biomass calculation process is implemented in `lambda_function.py` and follows these steps:

1. **Input Processing**: The function receives a multi-band GeoTIFF file from S3 containing at least two bands:
   - **Band 1 (NDVI)**: Normalized Difference Vegetation Index - a measure of vegetation health and density
   - **Band 2 (EVI)**: Enhanced Vegetation Index - an improved vegetation index that accounts for atmospheric conditions

2. **Data Extraction**:
   ```python
   ndvi_data = src.read(ndvi_band_idx)  # Reads NDVI band (default: band 1)
   evi_data = src.read(evi_band_idx)    # Reads EVI band (default: band 2)
   ```

3. **Feature Preparation**:
   - The NDVI and EVI arrays are flattened into 1D arrays
   - They are combined into a feature matrix where each row represents a pixel with `[NDVI, EVI]` values
   - The data is reshaped to `(height × width, 2)` for model prediction

4. **NoData Handling**:
   - The function identifies pixels with NoData values in either NDVI or EVI bands
   - Only valid pixels (those without NoData values) are processed
   - NoData pixels are preserved in the output with the original NoData value

5. **Biomass Prediction**:
   - A pre-trained scikit-learn model (loaded from S3) is used to predict biomass
   - The model takes the `[NDVI, EVI]` feature pairs as input
   - Predictions are made only for valid pixels: `predicted_biomass = model.predict(valid_input_data)`

6. **Output Generation**:
   - The predicted biomass values are mapped back to their original pixel positions
   - NoData pixels are filled with the NoData value (or -9999 if not specified)
   - A new single-band GeoTIFF is created with:
     - Data type: `float32`
     - Single band containing biomass estimates
     - Preserved geospatial metadata (CRS, transform, etc.)

### Key Functions

#### `calculate_biomass_on_tif(input_tif_bytes, model, ndvi_band_idx=1, evi_band_idx=2)`

This is the core function that performs the biomass calculation:

- **Input**: In-memory GeoTIFF bytes and a trained scikit-learn model
- **Process**: 
  - Extracts NDVI and EVI bands
  - Creates feature matrix `[NDVI, EVI]` for each pixel
  - Filters out NoData pixels
  - Applies ML model to predict biomass
  - Reconstructs output raster with predictions
- **Output**: In-memory GeoTIFF bytes containing biomass estimates

#### `load_model_from_s3(bucket, key)`

Loads the trained machine learning model from S3:
- Downloads the model file (typically a `.joblib` file)
- Deserializes it using `joblib.load()`
- Returns the model object ready for predictions

### Model Requirements

The model must be:
- Trained using scikit-learn
- Serialized using `joblib`
- Expecting input features as `[NDVI, EVI]` pairs
- Stored in an S3 bucket accessible by the Lambda function

## Deployment Overview

This Lambda function is deployed as a Docker container image to AWS Lambda, providing a scalable and maintainable serverless solution.

### Architecture

```
S3 Input Bucket → Lambda Function → S3 Output Bucket
                      ↓
                 S3 Model Bucket
```

### Deployment Process Summary

1. **Environment Setup**:
   - Python 3.10.5 virtual environment using `pyenv-virtualenv`
   - Required dependencies: `rasterio`, `numpy`, `pandas`, `scikit-learn`, `joblib`, `boto3`

2. **Docker Container Deployment**:
   - Build Docker image containing the Lambda function and dependencies
   - Push image to AWS Elastic Container Registry (ECR)
   - Create/update Lambda function using the container image

3. **Lambda Configuration**:
   - **Package Type**: Container Image
   - **Timeout**: Configurable (default: 30s, max: 900s)
   - **Memory**: Configurable (default: 512MB, range: 128MB-10240MB)
   - **IAM Role**: Requires permissions for S3 read/write operations

4. **Integration**:
   - Can be triggered via S3 events (automatic processing on file upload)
   - Can be invoked via API Gateway (HTTP/REST API)
   - Can be invoked directly via AWS CLI or SDK

### Key Deployment Commands

```bash
# Build and push Docker image
docker build -t ts_biomass_ndvi_lambda_image .
docker tag ts_biomass_ndvi_lambda_image:latest <ECR_URI>:latest
docker push <ECR_URI>:latest

# Create/Update Lambda function
aws lambda create-function \
    --function-name ts_biomass_ndvi_lambda \
    --package-type Image \
    --code ImageUri=<ECR_URI>:latest \
    --role <IAM_ROLE_ARN> \
    --timeout 120 \
    --memory-size 2048

# Update function code
aws lambda update-function-code \
    --function-name ts_biomass_ndvi_lambda \
    --image-uri <ECR_URI>:latest
```

### Testing

The function can be tested through multiple methods:

1. **Direct Lambda Invocation** (AWS CLI)
2. **API Gateway** (HTTP POST requests)
3. **S3 Event Triggers** (automatic on file upload)

For detailed testing procedures and examples, refer to `DEPLOYMENT.md`.

## Input/Output Format

### Input Event Structure

```json
{
  "Records": [
    {
      "s3": {
        "bucket": {
          "name": "input-bucket-name"
        },
        "object": {
          "key": "path/to/input_file.tif"
        }
      }
    }
  ],
  "custom_payload": "{\"output_bucket_name\": \"output-bucket\", \"model_bucket_name\": \"model-bucket\", \"model_key\": \"biomass_model.joblib\"}"
}
```

### Output

- **Location**: Same S3 bucket as specified in `output_bucket_name`
- **Filename**: `biomass_map_{original_filename}.tif`
- **Format**: Single-band GeoTIFF with float32 biomass estimates
- **Metadata**: Preserves geospatial information from input file

## Requirements

### Python Dependencies

```
rasterio
boto3
scikit-learn
numpy
pandas
joblib
```

### AWS Resources

- **S3 Buckets**: 
  - Input bucket (for GeoTIFF files)
  - Output bucket (for biomass maps)
  - Model bucket (for ML model)
- **Lambda Function**: Container-based deployment
- **IAM Role**: With S3 read/write permissions
- **ECR Repository**: For storing Docker images
- **API Gateway** (optional): For HTTP access

## Usage Example

### Via API Gateway

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
            "key": "input_image.tif"
          }
        }
      }
    ],
    "custom_payload": "{\"output_bucket_name\": \"tsbiomassmodeldata\", \"model_bucket_name\": \"tsbiomassmodeldata\"}"
  }' \
  https://<api-gateway-url>/predict_nvdi_tif
```

### Verify Output

```bash
# List generated biomass maps
aws s3 ls s3://output-bucket/ | grep biomass_map

# Download result
aws s3 cp s3://output-bucket/biomass_map_input_image.tif ~/Downloads/
```

## Error Handling

The function includes comprehensive error handling for:
- Invalid S3 event structure
- Missing configuration parameters
- Model loading failures
- File download/upload errors
- NoData scenarios (returns appropriate error response)
- Biomass calculation failures

All errors are logged to CloudWatch Logs for debugging.

## Performance Considerations

- **Memory**: Processing large GeoTIFF files may require increased memory allocation (up to 10GB)
- **Timeout**: Large files may need extended timeout (up to 15 minutes)
- **Processing**: All operations are performed in-memory to optimize performance
- **Scalability**: Lambda automatically scales to handle concurrent requests

## Related Documentation

For detailed deployment instructions, testing procedures, and configuration options, see `DEPLOYMENT.md`.

