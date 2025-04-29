# AWS Deployment Guide

This guide provides instructions for deploying the trading-strategy-backtester alongside a frontend application on AWS.

## Prerequisites

- AWS Account
- AWS CLI configured on your local machine
- Docker and Docker Compose installed on your local machine
- Both `trading-strategy-backtester` and `frontend` repositories cloned

## Deployment Options

There are several ways to deploy this application stack on AWS:

1. **Amazon EC2**: Traditional VM-based deployment
2. **Amazon ECS**: Docker container orchestration
3. **Amazon EKS**: Kubernetes-based deployment

This guide focuses on the EC2 approach (simplest) and ECS approach (more scalable).

## Option 1: EC2 Deployment

### 1. Launch an EC2 Instance

1. Go to the AWS Management Console and navigate to EC2
2. Click **Launch Instance**
3. Choose an Amazon Linux 2 AMI (or Ubuntu)
4. Select an instance type (recommended: t2.medium or larger)
5. Configure instance details as needed
6. Add storage (recommended: at least 20GB)
7. Configure security group to allow:
   - SSH (port 22) from your IP
   - HTTP (port 80) from anywhere
   - HTTPS (port 443) from anywhere
   - Custom TCP (port 3000) from anywhere (for the frontend)
8. Launch the instance and create/select a key pair

### 2. Connect to Your EC2 Instance

```bash
ssh -i /path/to/your-key.pem ec2-user@your-instance-public-dns
```

### 3. Install Docker and Docker Compose

For Amazon Linux 2:

```bash
# Update packages
sudo yum update -y

# Install Docker
sudo amazon-linux-extras install docker -y
sudo service docker start
sudo systemctl enable docker
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Log out and log back in to apply group changes
exit
# Reconnect using ssh
```

### 4. Upload Your Code to EC2

You can either use GitHub to clone your repositories or SCP to upload them:

```bash
# Create a project directory
mkdir -p ~/trading-app

# Clone your repositories
cd ~/trading-app
git clone <trading-strategy-backtester-repo-url> trading-strategy-backtester
git clone <frontend-repo-url> frontend
```

Alternatively, use SCP from your local machine:

```bash
# From your local machine
scp -i /path/to/your-key.pem -r /path/to/trading-strategy-backtester ec2-user@your-instance-public-dns:~/trading-app/
scp -i /path/to/your-key.pem -r /path/to/frontend ec2-user@your-instance-public-dns:~/trading-app/
```

### 5. Configure Docker Compose

Copy the parent docker-compose file to the project root:

```bash
cd ~/trading-app
cp trading-strategy-backtester/docker-compose.parent.yml docker-compose.yml
```

Modify the docker-compose.yml to set the correct ports and ensure frontend is accessible:

```bash
nano docker-compose.yml
```

Make sure your frontend service has the correct port mapping:

```yaml
frontend:
  build:
    context: ./frontend
    dockerfile: Dockerfile
  ports:
    - "80:3000"  # Map port 80 to container port 3000
  volumes:
    - shared-data:/app/frontend/public/data
  depends_on:
    - backtester
```

### 6. Start Your Application

```bash
cd ~/trading-app
docker-compose up -d
```

### 7. Access Your Application

Your frontend should now be accessible at:

```
http://your-instance-public-dns
```

To access the backtester logs or run commands:

```bash
# View logs
docker-compose logs backtester

# Run a backtest
docker-compose exec backtester python src/workflows/cli.py --workflow simple --strategy MACrossover --tickers AAPL
```

## Option 2: Amazon ECS Deployment

### 1. Create an ECR Repository for Your Images

1. Go to the AWS Management Console and navigate to ECR
2. Create two repositories: one for the backtester and one for the frontend

### 2. Push Your Docker Images to ECR

```bash
# Log in to ECR
aws ecr get-login-password --region your-region | docker login --username AWS --password-stdin your-account-id.dkr.ecr.your-region.amazonaws.com

# Build and tag your images
cd ~/path/to/trading-strategy-backtester
docker build -t your-account-id.dkr.ecr.your-region.amazonaws.com/backtester:latest .

cd ~/path/to/frontend
docker build -t your-account-id.dkr.ecr.your-region.amazonaws.com/frontend:latest .

# Push the images
docker push your-account-id.dkr.ecr.your-region.amazonaws.com/backtester:latest
docker push your-account-id.dkr.ecr.your-region.amazonaws.com/frontend:latest
```

### 3. Create ECS Task Definitions

1. Go to the AWS Management Console and navigate to ECS
2. Create Task Definitions for:
   - Backtester service
   - Frontend service
   - Include the shared volume configuration

Example task definition (JSON format):

```json
{
  "family": "trading-app",
  "networkMode": "awsvpc",
  "executionRoleArn": "arn:aws:iam::your-account-id:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "backtester",
      "image": "your-account-id.dkr.ecr.your-region.amazonaws.com/backtester:latest",
      "essential": true,
      "environment": [
        { "name": "BASE_DIR", "value": "/app/trading-strategy-backtester" }
      ],
      "mountPoints": [
        {
          "sourceVolume": "data-volume",
          "containerPath": "/app/trading-strategy-backtester/output/shared",
          "readOnly": false
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/trading-app",
          "awslogs-region": "your-region",
          "awslogs-stream-prefix": "backtester"
        }
      }
    },
    {
      "name": "frontend",
      "image": "your-account-id.dkr.ecr.your-region.amazonaws.com/frontend:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 3000,
          "hostPort": 3000,
          "protocol": "tcp"
        }
      ],
      "mountPoints": [
        {
          "sourceVolume": "data-volume",
          "containerPath": "/app/frontend/public/data",
          "readOnly": true
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/trading-app",
          "awslogs-region": "your-region",
          "awslogs-stream-prefix": "frontend"
        }
      }
    }
  ],
  "volumes": [
    {
      "name": "data-volume",
      "dockerVolumeConfiguration": {
        "scope": "shared",
        "autoprovision": true,
        "driver": "local"
      }
    }
  ],
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048"
}
```

### 4. Create an ECS Cluster

1. Go to the AWS Management Console and navigate to ECS
2. Create a new cluster, choosing either EC2 or Fargate launch type

### 5. Create ECS Services

1. In your ECS cluster, create a service using your task definition
2. Configure the service with appropriate networking settings
3. Set up load balancing if needed
4. Launch the service

### 6. Access Your Application

Your frontend should now be accessible through the DNS name of your load balancer or the public IP of your ECS instance.

## Common AWS Deployment Considerations

### 1. Persistent Storage

For both EC2 and ECS deployments, consider using:

- **Amazon EFS**: For shared persistent storage (better for production)
- **Amazon S3**: For storing backtest results and data files

Update your docker-compose.yml or ECS task definition to use EFS:

```yaml
volumes:
  shared-data:
    driver: efs
    driver_opts:
      fs-id: fs-xxxxxxxx
      rootdir: /path/in/efs
```

### 2. Database Integration

If your frontend needs a database:

1. Create an Amazon RDS instance 
2. Configure your frontend to connect to the RDS endpoint
3. Make sure security groups allow the connection

### 3. Load Balancing and Auto Scaling

For production deployments:

1. Set up an Application Load Balancer
2. Configure auto-scaling based on CPU/memory usage
3. Set up target groups for your services

### 4. Custom Domain and HTTPS

1. Register a domain in Route 53 or use an existing domain
2. Set up an SSL certificate using AWS Certificate Manager
3. Configure your load balancer to use HTTPS

### 5. CI/CD Pipeline

Set up a CI/CD pipeline using AWS CodePipeline:

1. Source: CodeCommit, GitHub, or Bitbucket
2. Build: CodeBuild to build Docker images
3. Deploy: CodeDeploy to deploy to ECS or EC2

## Using the Application

### Running Backtests

You can run backtests on your deployed application:

```bash
# For EC2 deployment
docker-compose exec backtester python src/workflows/cli.py --workflow simple --strategy MACrossover --tickers AAPL

# For ECS deployment, use AWS ECS Execute Command
aws ecs execute-command --cluster your-cluster --task task-id --container backtester --command "python src/workflows/cli.py --workflow simple --strategy MACrossover --tickers AAPL" --interactive
```

### Viewing Results

Results will be stored in the shared volume and should be accessible to the frontend application. You can also download results from the EC2 instance or set up S3 syncing to persist them.

### Monitoring the Application

- Use CloudWatch to monitor your containers and EC2 instances
- Set up CloudWatch Logs to centralize log collection
- Set up CloudWatch Alarms for critical metrics

## Troubleshooting

### Connection Issues
- Check security groups to ensure ports are open
- Verify network ACLs and routing tables

### Container Issues
- Check Docker logs: `docker-compose logs` or CloudWatch Logs
- Ensure volumes are mounted correctly
- Verify environment variables are set properly

### Permission Issues
- Ensure proper IAM roles are attached to EC2 instances or ECS tasks
- Check file permissions in shared volumes

## Cost Optimization

- Use Spot Instances for non-critical workloads
- Use Fargate for sporadic workloads
- Set up auto-scaling to scale down during low usage periods
- Consider using Application Auto Scaling to right-size your applications