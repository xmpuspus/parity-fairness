import boto3
import json

def dump_to_aws_s3(access_key_file, 
                   bucket, 
                   fn_to_upload, 
                   target_object_fn):
    
    """
    Dump any data file into s3 from jupyter notebook.
    
    Parameters:
    access_key_file (json): json file of aws access keys
    bucket (string): Name of s3 bucket
    fn_to_upload (string): Filename of the saved file to be uploaded.
    target_object_fn (string): Filename of the saved s3 object.
    
    Returns:
    Executes object dump to your s3 bucket with a prompt of the target filename.
    """
    
    # Access key file has to be dictionary/unnested json format
    with open(access_key_file) as json_file:
      access = json.load(json_file)

    # Configure aws access
    ACCESS_KEY_ID = access['ACCESS_KEY_ID']
    SECRET_ACCESS_KEY = access['SECRET_ACCESS_KEY']
    BUCKET = bucket

    # Initialize s3
    s3 = boto3.client('s3', 
                      aws_access_key_id=ACCESS_KEY_ID, 
                      aws_secret_access_key=SECRET_ACCESS_KEY)
    
    s3.upload_file(fn_to_upload, 
                BUCKET, 
                target_object_fn)
  
    print(f'Done! Check s3://{BUCKET}/{target_object_fn}')
