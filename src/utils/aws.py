import io
import json
import pandas as pd
import boto3

# Initialize the S3 client
s3_client = boto3.client('s3')

def read_csv_io(bucket_name: str, file_key: str):
    csv_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    csv_data = csv_obj['Body'].read().decode('utf-8')
    df = pd.read_csv(io.StringIO(csv_data))
    return df

def read_json(bucket_name: str, file_key: str):
    json_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    json_data = json_obj['Body'].read().decode('utf-8')
    json_content = json.loads(json_data)
    return json_content

def save_json_to_s3(json_content: dict, bucket_name: str, file_key: str):
    json_data = json.dumps(json_content).encode('utf-8')
    s3_client.upload_fileobj(io.BytesIO(json_data), bucket_name, file_key)

async def save_json_to_s3_async(json_content: dict, bucket_name: str, file_key: str):
    json_data = json.dumps(json_content).encode('utf-8')
    s3_client.upload_fileobj(io.BytesIO(json_data), bucket_name, file_key)

def save_local_to_s3(local_file_path: str, bucket_name: str, file_key: str):
    with open(local_file_path, 'rb') as file_data:
        s3_client.upload_fileobj(file_data, bucket_name, file_key)

def save_df_to_s3(df: pd.DataFrame, bucket_name: str, file_key: str):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    s3_client.put_object(Body=csv_buffer.getvalue().encode('utf-8'), Bucket=bucket_name, Key=file_key)