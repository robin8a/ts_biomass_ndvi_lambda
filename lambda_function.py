import json
import os
import io
import joblib
import rasterio
import numpy as np
import boto3

# Initialize boto3 client
s3_client = boto3.client('s3')

# --- Helper Functions (unchanged) ---

def load_model_from_s3(bucket, key):
    """Loads the trained model from S3 into memory."""
    try:
        model_object = s3_client.get_object(Bucket=bucket, Key=key)
        model_bytes = model_object['Body'].read()
        model = joblib.load(io.BytesIO(model_bytes))
        print(f"Model '{key}' loaded from S3 bucket '{bucket}'.")
        return model
    except Exception as e:
        print(f"Error loading model from S3: {e}")
        raise e

def calculate_biomass_on_tif(input_tif_bytes, model, ndvi_band_idx=1, evi_band_idx=2):
    """
    Reads a multi-band .tif from in-memory bytes, calculates biomass for each pixel,
    and returns the result as a new single-band .tif in memory.

    Args:
        input_tif_bytes (io.BytesIO): In-memory bytes of the input multi-band .tif.
        model: The trained scikit-learn model.
        ndvi_band_idx (int): The 1-based index of the NDVI band.
        evi_band_idx (int): The 1-based index of the EVI band.
    """
    print("Starting biomass calculation...")
    with rasterio.open(input_tif_bytes) as src:
        profile = src.profile

        # Read the specified bands
        ndvi_data = src.read(ndvi_band_idx)
        evi_data = src.read(evi_band_idx)

        # Get the dimensions and reshape for prediction
        height, width = ndvi_data.shape
        input_data = np.vstack((ndvi_data.flatten(), evi_data.flatten())).T

        # Handle nodata values
        nodata_value = profile.get('nodata')
        if nodata_value is not None:
            nodata_mask = (ndvi_data == nodata_value) | (evi_data == nodata_value)
            valid_indices = ~nodata_mask.flatten()
            valid_input_data = input_data[valid_indices]
        else:
            valid_input_data = input_data
            valid_indices = np.full(input_data.shape[0], True)

        if valid_input_data.size == 0:
            print("No valid data found in the specified bands. Aborting.")
            return None

        print("Predicting biomass for valid pixels...")
        predicted_biomass = model.predict(valid_input_data)

        # Create an output array and fill with nodata
        output_biomass = np.full(height * width, nodata_value if nodata_value is not None else -9999, dtype=np.float32)
        output_biomass[valid_indices] = predicted_biomass
        output_biomass = output_biomass.reshape(height, width)

        # Update the profile for the new output file
        profile.update(
            dtype=rasterio.float32,
            count=1,
            driver='GTiff',
            nodata=nodata_value if nodata_value is not None else -9999
        )

        # Write the new raster file to a BytesIO object
        output_buffer = io.BytesIO()
        with rasterio.open(output_buffer, 'w', **profile) as dst:
            dst.write(output_biomass, 1)

        output_buffer.seek(0)
        print("Biomass calculation complete.")
        return output_buffer


def lambda_handler(event, context):
    """
    AWS Lambda function handler triggered by an S3 object creation event.
    The output bucket name and model bucket name are now passed via the event payload.
    """
    # Parse the S3 event to get the bucket and key of the uploaded file
    try:
        input_bucket_name = event['Records'][0]['s3']['bucket']['name']
        object_key = event['Records'][0]['s3']['object']['key']
    except KeyError as e:
        print(f"Error parsing S3 event: {e}")
        return {
            'statusCode': 400,
            'body': json.dumps('Error: Invalid S3 event structure.')
        }

    # Extract the bucket names from the event payload
    try:
        # Assuming the bucket names are provided in a 'configuration' key
        # You may need to adjust this based on how your event is structured.
        config = json.loads(event.get('custom_payload', '{}'))
        print(f"Extracted configuration: {config}")
        output_bucket_name = config.get('output_bucket_name')
        model_bucket_name = config.get('model_bucket_name')
        model_key = config.get('model_key', 'biomass_model.joblib')

        if not output_bucket_name or not model_bucket_name:
            raise ValueError("Output and model bucket names must be provided in the event.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error extracting bucket names from event: {e}")
        return {
            'statusCode': 400,
            'body': json.dumps('Error: Missing or invalid bucket name parameters in event.')
        }

    print(f"Processing file: {object_key} from input bucket: {input_bucket_name}")
    print(f"Model will be loaded from: {model_bucket_name}/{model_key}")
    print(f"Output will be saved to: {output_bucket_name}")

    # 1. Load the trained model
    try:
        model = load_model_from_s3(model_bucket_name, model_key)
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps(f'Failed to load model: {e}')
        }

    # 2. Download the input .tif from S3 into memory
    try:
        input_object = s3_client.get_object(Bucket=input_bucket_name, Key=object_key)
        input_tif_bytes = io.BytesIO(input_object['Body'].read())
        print(f"Downloaded {object_key} from S3.")
    except Exception as e:
        print(f"Error downloading file: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error downloading {object_key}: {e}')
        }

    # 3. Perform the biomass calculation
    try:
        output_tif_bytes = calculate_biomass_on_tif(input_tif_bytes, model)
    except Exception as e:
        print(f"Error during biomass calculation: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error calculating biomass: {e}')
        }

    if output_tif_bytes is None:
        return {
            'statusCode': 500,
            'body': json.dumps('Biomass calculation failed due to no valid data.')
        }

    # 4. Upload the result to the output S3 bucket
    try:
        output_key = f"biomass_map_{os.path.basename(object_key)}"
        s3_client.put_object(Bucket=output_bucket_name, Key=output_key, Body=output_tif_bytes)
        print(f"Successfully uploaded biomass map to s3://{output_bucket_name}/{output_key}")
    except Exception as e:
        print(f"Error uploading result: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error uploading result to S3: {e}')
        }

    return {
        'statusCode': 200,
        'body': json.dumps(f"Biomass calculation for '{object_key}' is complete.")
    }