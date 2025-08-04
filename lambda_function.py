import json
import os
import io
import joblib
import rasterio
import numpy as np
import boto3
import matplotlib.pyplot as plt
from PIL import Image

# Initialize boto3 client
s3_client = boto3.client('s3')

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
    and returns both the output .tif bytes and the biomass numpy array.
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
            return None, None

        print("Predicting biomass for valid pixels...")
        predicted_biomass = model.predict(valid_input_data)

        # Create an output array and fill with nodata
        output_biomass_array = np.full(height * width, nodata_value if nodata_value is not None else -9999, dtype=np.float32)
        output_biomass_array[valid_indices] = predicted_biomass
        output_biomass_array = output_biomass_array.reshape(height, width)

        # Update the profile for the new output file
        profile.update(
            dtype=rasterio.float32,
            count=1,
            driver='GTiff',
            nodata=nodata_value if nodata_value is not None else -9999
        )

        # Write the new raster file to a BytesIO object
        output_tif_buffer = io.BytesIO()
        with rasterio.open(output_tif_buffer, 'w', **profile) as dst:
            dst.write(output_biomass_array, 1)

        output_tif_buffer.seek(0)
        print("Biomass calculation complete.")
        return output_tif_buffer, output_biomass_array

def create_png_from_biomass_data(biomass_array):
    """
    Converts a biomass numpy array to a PNG image in memory using matplotlib.
    """
    print("Creating PNG image from biomass data...")
    png_buffer = io.BytesIO()
    
    # Use matplotlib to create the image with a colormap
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # Replace nodata values with NaN so they are not plotted
    biomass_array[biomass_array == -9999] = np.nan
    img = ax.imshow(biomass_array, cmap='viridis')
    ax.axis('off')
    fig.colorbar(img, ax=ax, label='Biomass')

    # Save the plot to the buffer
    fig.savefig(png_buffer, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    png_buffer.seek(0)
    print("PNG image created.")
    return png_buffer

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
        config = json.loads(event.get('custom_payload', '{}'))
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

    # 3. Perform the biomass calculation and get both the tif bytes and the numpy array
    try:
        output_tif_bytes, biomass_array = calculate_biomass_on_tif(input_tif_bytes, model)
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

    # 4. Upload the resulting .tif to the output S3 bucket
    output_tif_key = f"biomass_map_{os.path.basename(object_key)}"
    try:
        s3_client.put_object(Bucket=output_bucket_name, Key=output_tif_key, Body=output_tif_bytes)
        print(f"Successfully uploaded biomass map (.tif) to s3://{output_bucket_name}/{output_tif_key}")
    except Exception as e:
        print(f"Error uploading result (.tif): {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error uploading biomass map (.tif) to S3: {e}')
        }
    
    # 5. Create and upload the resulting .png to the output S3 bucket
    output_png_key = f"biomass_map_{os.path.splitext(os.path.basename(object_key))[0]}.png"
    try:
        png_bytes = create_png_from_biomass_data(biomass_array)
        s3_client.put_object(Bucket=output_bucket_name, Key=output_png_key, Body=png_bytes)
        print(f"Successfully uploaded biomass map (.png) to s3://{output_bucket_name}/{output_png_key}")
    except Exception as e:
        print(f"Error creating/uploading result (.png): {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error creating or uploading biomass map (.png) to S3: {e}')
        }

    # Construct the S3 URLs for the response
    s3_url_base = f"https://{output_bucket_name}.s3.amazonaws.com/"
    tif_url = f"{s3_url_base}{output_tif_key}"
    png_url = f"{s3_url_base}{output_png_key}"

    return {
        'statusCode': 200,
        'body': json.dumps({
            "message": f"Biomass calculation for '{object_key}' is complete.",
            "output_files": {
                "biomass_map_tif": tif_url,
                "biomass_map_png": png_url
            }
        })
    }