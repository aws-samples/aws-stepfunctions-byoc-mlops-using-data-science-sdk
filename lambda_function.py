import boto3
import os
import logging
import time

# with zipfile.ZipFile('/tmp/source.zip', 'w') as myzip:
#     myzip.write('Dockerfile')
#     myzip.write('buildspec.yml')

codebuild_client = boto3.client('codebuild')
s3_client = boto3.client('s3')
ecr_client = boto3.client('ecr')

rolearn = os.environ['rolearn']
region = boto3.session.Session().region_name
acct_id = boto3.client('sts').get_caller_identity().get('Account')
reponame = os.environ['IMAGE_REPO']
imagename = os.environ['IMAGE_TAG']
bucket = os.environ['bucket']
project = os.environ['project']
dockerfilename = os.environ['dockerfilename']
folder_name = os.environ['trainscripts']

print("building {}:{}".format(reponame,imagename))
print("------------------------------------------")

from botocore.exceptions import ClientError


def upload_folder(bucket, folder_name):
    for path, subdirs, files in os.walk(folder_name):
            path = path.replace("\\","/")
            directory_name = path.replace(folder_name,"")
            for file in files:
                print(os.path.join(path, file))
                s3_client.upload_file(os.path.join(path, file), bucket, file)
                

def lambda_handler(event, context):
    #try deleting existing repository 
    if os.environ['repositoryexists']=='yes':
        print("Existing repository with name {}, not recreating".format(project))
        print(ecr_client.describe_repositories(repositoryNames=[reponame]))
    else:
        print("No existing repository with that name. Building as usual")
        response = ecr_client.create_repository(repositoryName=reponame)
    
    
    #try deleting existing project
    try:
        codebuild_client.delete_project(name=project)
        print("Existing project with name {}, deleting and rebuilding".format(project))
        print("deleted...")
    except:
        print("No existing project with that name. Building as usual")
    
    print(boto3.client('sts').get_caller_identity().get('Account') + ' starting codebuild...')
    
    # Upload to S3
    print("Uploading source files to S3 bucket :"+bucket)
    print(os.environ['dockerfilename'])
    for file_name in ['buildspec.yml', dockerfilename]:#,'/tmp/source.zip']:
        try:
            response = s3_client.upload_file(file_name, bucket, 'Dockerfile' if 'Dockerfile' in file_name else file_name)
            print("Pushed "+ file_name)
        except ClientError as e:
            logging.error(e)
        try:
            upload_folder(bucket, folder_name)
            print("Pushed " + folder_name)
        except ClientError as e:
            logging.error(e)

    # Send to codebuild
    response = codebuild_client.list_projects()
    print(response)
    if project in response['projects']:
        print("Project already created")
    else:
        print("creating project")
        print('s3://'+bucket+'/')
        response = codebuild_client.create_project(name=project, description='lambda to docker build',
        source={
            'type': 'S3',
            'location': bucket+"/",
            'buildspec': 'buildspec.yml'},
        artifacts={'type': 'NO_ARTIFACTS'},
        environment={
            'type': 'LINUX_CONTAINER',
            'image': 'aws/codebuild/standard:2.0',
            'computeType': 'BUILD_GENERAL1_MEDIUM',
            "privilegedMode": True,
            'environmentVariables': [
            {
                "name": "AWS_DEFAULT_REGION",
                "value": region
                
            },
            {
                "name": "AWS_ACCOUNT_ID",
                "value": acct_id
                
            },
            {
                'name': 'IMAGE_REPO',
                'value': reponame,
                'type': 'PLAINTEXT'
            },
            {
                'name': 'IMAGE_TAG',
                'value': imagename,
                'type': 'PLAINTEXT'
            }],
        },
        serviceRole=rolearn,
        )
            
    print('build project')
    
    response = codebuild_client.start_build(projectName=project)
    buildid = response['build']['id']
    status = 'CHECK_LOGS'
    if(os.environ['waitforbuild']=='yes'):
        status = 'IN_PROGRESS'
        c=1
        while status == 'IN_PROGRESS':
            print("-", end = '')
            time.sleep(5)
            c=c+1
            response = codebuild_client.batch_get_builds(ids=[buildid])
        
            for b in response['builds']:
                if b['id'] == buildid:
                    status = b['buildStatus']
        
        print("!")
        print(status)
        
    return {
        'statusCode': 200,
        'status': status,
        'buildtime':c*5,
        'buildid':buildid,
        'repo':reponame,
        'image':imagename
    }


