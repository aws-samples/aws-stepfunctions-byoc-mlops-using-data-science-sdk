# Build, Train and Deploy your own Mask R-CNN container to Amazon SageMaker using AWS StepFunctions Data Science SDK. 

![](media/workflow.png)


This workshop demonstrates how to use the StepFunction Data Science SDK to build train and deploy your own container in Amazon
SageMaker using only Python Code. This enables data scientists to develop Continuous Integration/Continuous Delivery (CI/CD) pipelines
into their workflow. 

The overall flow of this workshop is as follows:

1/ Upload your code to the Lambda console <br/>
2/ Use StepFunctions pipeline to kick off the Lambda function which in-turn will launch a CodeBuild job to build your Mask R-CNN
Docker container with your custom code <br/>
3/ CodeBuild will upload the Docker container to Amazon ECR for your use.<br/>
4/ The StepFunctions Training pipeline will pick up this Docker container, train the model and deploy it. <br/>

**Caution** Note that in order to train a Mask R-CNN model, we use an ml.p3.2xlarge instance with a training time of roughly 320 seconds. 
This will incur a cost of $0.38 for training the model.


# Step 1: Deploy this CloudFormation template

This Cloudformation template creates a Lambda function with a sample Dockerfile that launches a Codebuild job
to build a Sagemaker specific container. 

Launch CloudFormation stack in us-east-1: [![button](media/cloudformation-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/create/template?stackName=lambda-docker-build&templateURL=https://lambda-ml-layers.s3.amazonaws.com/lambda-sm-build.yaml)

SageMaker Containers gives you tools to create SageMaker-compatible Docker containers, 
and has additional tools for letting you create Frameworks (SageMaker-compatible Docker containers that can run arbitrary Python or shell scripts). 
Currently, this library is used by the following containers: TensorFlow Script Mode, MXNet, PyTorch, Chainer, and Scikit-learn.

# Step 2: Clone this repo to your home directory

#Step 3: Upload your code

Next in a Terminal, navigate to the folder containing the git repo you just cloned.

Run the following command:

```bash
zip -r lambda.zip Dockerfile buildspec.yml lambda_function.py mask_r_cnn/*
```

# Step 3: Modify the Lambda function

Next, navigate to the Lambda function you just created, and in the Function Code section, for Code entry type, select: **upload a .zip file**.

Upload your file "lambda.zip" to this code and click Save.

Next, scroll down to Environment Variables and replace the following Variables with the ones shown in the diagram below.

1. IMAGE_REPO <br/>
2. IMAGE_TAG <br/>
3. trainscripts <br/>

![](media/lambdaenv.png)

Next, scroll down to Basic Settings and update the Lambda Timeout to 15 minutes and the memory to 1024MB. While this is overkill, this will avoid any OOM or timeout issues for larger code files. 

# Step 4: Run the Jupyter Notebook.

Create a new SageMaker notebook and upload the notebook *StepFunctions_BYOC_Workflow.ipynb*. 

To Download the dataset, go to the link here: https://www.cis.upenn.edu/~jshi/ped_html/.

Unzip the dataset files and upload the data folder to your SageMaker notebook environment. 

In order to run the workshop, you will need to ensure that Amazon SageMaker can call AWS StepFunctions, and that StepFunctions can call Model Training, Creation, and Deployment on behalf of SageMaker. To ensure the correct IAM Persmissions, refer to the **Setup** section of the following notebook: https://github.com/awslabs/amazon-sagemaker-examples/blob/master/step-functions-data-science-sdk/machine_learning_workflow_abalone/machine_learning_workflow_abalone.ipynb.

Once the appropriate setup is complete, run through the *StepFunctions_BYOC_Workflow.ipynb*. Open up the StepFunctions Console to watch the individual steps in the graph getting executed.

![](media/SFgraph.png)

At the end of this workshop, you should have a deployed SageMaker endpoint that you can use to call inferences on your model.

# Next Steps

While the example we demonstrate here works for Mask R-CNN, with simple modificiations, you can use this to bring your own algorithm
container to Amazon SageMaker. For precise steps on how to modify your container code to work on Amazon SageMaker, we refer you
to this Github repo: https://github.com/awslabs/amazon-sagemaker-examples/tree/master/advanced_functionality/scikit_bring_your_own.

This repo demonstrates how to package your training and inference code, and Dockerfile in a format that is compatible with SageMaker. For the Mask R-CNN workshop here, this has already been done for you. 

Enjoy building your own CI/CD pipelines with Amazon SageMaker, AWS Lambda, AWS StepFunctions
