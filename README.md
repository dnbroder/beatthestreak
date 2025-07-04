# BeatTheStreak

Documentation to come soon :)

(Documentation will contain system diagram, instructions for execution and how to interpret results.)

1. CICD Role (mlb-cicd-role) must be deployed prior to running any deployment of CloudFormation resources via GitHub. IAM policy for this role can be found here:

2. ECR Registry must be deployed via CLI.

3. Initally, ecs_stack.yaml must be manually deployed in CloudFormation.

4. Right now, CloudFormation stack must be torn down and reconstructed whenever there are changes.