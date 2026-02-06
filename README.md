# BeatTheStreak

Documentation to come soon, like real soon:)

(Documentation will contain system diagram, instructions for execution and how to interpret results.)

1. CICD Role (mlb-cicd-role) must be deployed prior to running any deployment of CloudFormation resources via GitHub. IAM policy for this role can be found here:

AWSTemplateFormatVersion: '2010-09-09'
Description: IAM Role for GitHub Actions CI/CD Deployments

Resources:
  CICDRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: mlb-cicd-role
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Federated: arn:aws:iam::{YourAccountNumberGoesHere}:oidc-provider/token.actions.githubusercontent.com
            Action: sts:AssumeRoleWithWebIdentity
            Condition:
              StringEquals:
                token.actions.githubusercontent.com:aud: sts.amazonaws.com
                token.actions.githubusercontent.com:sub: repo:dnbroder/beatthestreak:ref:refs/heads/main
      Policies:
        - PolicyName: CICDPipelinePolicy
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - cloudformation:CreateStack
                  - cloudformation:UpdateStack
                  - cloudformation:DescribeStacks
                  - cloudformation:DescribeStackEvents
                  - cloudformation:GetTemplateSummary
                  - cloudformation:ValidateTemplate
                  - cloudformation:CreateChangeSet
                  - cloudformation:ExecuteChangeSet
                  - cloudformation:DeleteStack
                  - iam:GetRole
                  - iam:PassRole
                  - s3:PutObject
                  - s3:GetObject
                  - s3:ListBucket
                  - kms:GenerateDataKey
                  - kms:Decrypt
                  - ecr:GetAuthorizationToken
                  - ecr:InitiateLayerUpload
                  - ecr:UploadLayerPart
                  - ecr:CompleteLayerUpload
                  - ecr:BatchCheckLayerAvailability
                  - ecr:GetDownloadUrlForLayer
                  - ecr:PutImage
                  - ecs:UpdateService
                Resource: "*"