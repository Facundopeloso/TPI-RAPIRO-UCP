# ============================================================
# terraform/main.tf
# Infraestructura AWS para RAPIRO Guardián de Distracción
# Región: us-east-1
# ============================================================

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "rapiro-terraform-state"
    key            = "rapiro-guardian/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "rapiro-terraform-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region
}

# ---------------------------------------------------------------------------
# AWS IoT Core
# ---------------------------------------------------------------------------

resource "aws_iot_thing" "rapiro" {
  name = "rapiro-guardian-${var.environment}"
}

resource "aws_iot_certificate" "rapiro" {
  active = true
}

resource "aws_iot_policy" "rapiro_publish" {
  name = "rapiro-publish-policy-${var.environment}"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["iot:Publish", "iot:Connect"]
      Resource = "arn:aws:iot:${var.aws_region}:*:topic/${var.iot_topic}"
    }]
  })
}

resource "aws_iot_policy_attachment" "rapiro" {
  policy = aws_iot_policy.rapiro_publish.name
  target = aws_iot_certificate.rapiro.arn
}

resource "aws_iot_topic_rule" "to_lambda" {
  name        = "rapiro_events_to_lambda_${var.environment}"
  enabled     = true
  sql         = "SELECT * FROM '${var.iot_topic}'"
  sql_version = "2016-03-23"

  lambda {
    function_arn = aws_lambda_function.event_processor.arn
  }
}

# ---------------------------------------------------------------------------
# DynamoDB
# ---------------------------------------------------------------------------

resource "aws_dynamodb_table" "sessions" {
  name           = "rapiro_sessions_${var.environment}"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "session_id"
  range_key      = "timestamp"

  attribute {
    name = "session_id"
    type = "S"
  }

  attribute {
    name = "timestamp"
    type = "S"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }

  tags = var.common_tags
}

# ---------------------------------------------------------------------------
# S3 — Documentos de usuario
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "documents" {
  bucket = "rapiro-user-documents-${var.environment}-${data.aws_caller_identity.current.account_id}"
  tags   = var.common_tags
}

resource "aws_s3_bucket_versioning" "documents" {
  bucket = aws_s3_bucket.documents.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "documents_lifecycle" {
  bucket = aws_s3_bucket.documents.id

  rule {
    id     = "expire-old-documents"
    status = "Enabled"
    expiration {
      days = 90
    }
  }
}

# ---------------------------------------------------------------------------
# IAM Role para Lambda
# ---------------------------------------------------------------------------

resource "aws_iam_role" "lambda_role" {
  name = "rapiro-lambda-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "rapiro-lambda-policy-${var.environment}"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:Query"]
        Resource = aws_dynamodb_table.sessions.arn
      },
      {
        Effect   = "Allow"
        Action   = ["sns:Publish"]
        Resource = aws_sns_topic.alerts.arn
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# ---------------------------------------------------------------------------
# Lambda — Procesador de eventos
# ---------------------------------------------------------------------------

resource "aws_lambda_function" "event_processor" {
  filename         = "lambda/event_processor.zip"
  function_name    = "rapiro-event-processor-${var.environment}"
  role             = aws_iam_role.lambda_role.arn
  handler          = "handler.lambda_handler"
  runtime          = "python3.11"
  source_code_hash = filebase64sha256("lambda/event_processor.zip")
  timeout          = 30

  environment {
    variables = {
      DYNAMODB_TABLE    = aws_dynamodb_table.sessions.name
      SNS_TOPIC_ARN     = aws_sns_topic.alerts.arn
      ABSENCE_THRESHOLD = "300"
    }
  }

  tags = var.common_tags
}

resource "aws_lambda_permission" "iot_invoke" {
  statement_id  = "AllowIoTInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.event_processor.function_name
  principal     = "iot.amazonaws.com"
  source_arn    = aws_iot_topic_rule.to_lambda.arn
}

# ---------------------------------------------------------------------------
# SNS — Alertas
# ---------------------------------------------------------------------------

resource "aws_sns_topic" "alerts" {
  name = "rapiro-alerts-${var.environment}"
  tags = var.common_tags
}

resource "aws_sns_topic_subscription" "email_alert" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# ---------------------------------------------------------------------------
# CloudWatch — Alarmas
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "rapiro-lambda-errors-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  alarm_description   = "Lambda errors > 5 en 5 minutos"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    FunctionName = aws_lambda_function.event_processor.function_name
  }
}

resource "aws_cloudwatch_log_group" "event_processor" {
  name              = "/aws/lambda/${aws_lambda_function.event_processor.function_name}"
  retention_in_days = 14
  tags              = var.common_tags
}

# ---------------------------------------------------------------------------
# API Gateway — Dashboard
# ---------------------------------------------------------------------------

resource "aws_apigatewayv2_api" "dashboard" {
  name          = "rapiro-dashboard-api-${var.environment}"
  protocol_type = "HTTP"
  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST", "OPTIONS"]
    allow_headers = ["Content-Type", "Authorization"]
  }
  tags = var.common_tags
}

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

data "aws_caller_identity" "current" {}
