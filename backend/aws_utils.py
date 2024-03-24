"""aws util functions"""
import boto3
from botocore.exceptions import NoCredentialsError


def upload_file_to_s3(file_name, bucket_name, object_name=None):
    """
    Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket_name: Bucket to upload to
    :param object_name: S3 object name. If not specified, file_name is used
    :return: True if file was uploaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client("s3")
    try:
        s3_client.upload_file(
            file_name, bucket_name, object_name, ExtraArgs={"ContentType": "audio/mpeg"}
        )
    except NoCredentialsError:
        print("Credentials not available")
        return False
    return True


def create_presigned_url(bucket_name, object_name, expiration=3600):
    """
    Generate a presigned URL to share an S3 object

    :param bucket_name: string
    :param object_name: string
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Presigned URL as string. If error, returns None.
    """
    s3_client = boto3.client("s3")
    try:
        response = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_name},
            ExpiresIn=expiration,
        )
    except NoCredentialsError:
        print("Credentials not available")
        return None

    return response


# # Example usage
# bucket_name = 'YOUR_BUCKET_NAME'
# file_name = 'YOUR_FILE_NAME'
# object_name = 'YOUR_OBJECT_NAME'

# # Upload file
# if upload_file_to_s3(file_name, bucket_name, object_name):
#     print("Upload successful")
#     # Generate presigned URL
#     url = create_presigned_url(bucket_name, object_name)
#     if url:
#         print("Presigned URL: ", url)
#     else:
#         print("Could not generate presigned URL")
# else:
#     print("Upload failed")
