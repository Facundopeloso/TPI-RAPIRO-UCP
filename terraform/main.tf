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
    region         = "sa-east-1"
    use_lockfile   = true
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
    filter {}
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
      },
      {
        Effect   = "Allow"
        Action   = ["polly:SynthesizeSpeech"]
        Resource = "*"
      }
    ]
  })
}

# ---------------------------------------------------------------------------
# Lambda — Procesador de eventos
# ---------------------------------------------------------------------------

resource "aws_lambda_function" "event_processor" {
  filename         = "../lambda/event_processor.zip"
  function_name    = "rapiro-event-processor-${var.environment}"
  role             = aws_iam_role.lambda_role.arn
  handler          = "handler.lambda_handler"
  runtime          = "python3.11"
  source_code_hash = filebase64sha256("../lambda/event_processor.zip")
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

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# ---------------------------------------------------------------------------
# RED — VPC, Internet Gateway, Subred, Tabla de rutas
# ---------------------------------------------------------------------------

resource "aws_vpc" "rapiro_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  tags                 = merge(var.common_tags, { Name = "rapiro-vpc" })
}

resource "aws_internet_gateway" "rapiro_igw" {
  vpc_id = aws_vpc.rapiro_vpc.id
  tags   = merge(var.common_tags, { Name = "rapiro-igw" })
}

resource "aws_subnet" "rapiro_subnet" {
  vpc_id                  = aws_vpc.rapiro_vpc.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true
  tags                    = merge(var.common_tags, { Name = "rapiro-subnet" })
}

resource "aws_route_table" "rapiro_rt" {
  vpc_id = aws_vpc.rapiro_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.rapiro_igw.id
  }

  tags = merge(var.common_tags, { Name = "rapiro-rt" })
}

resource "aws_route_table_association" "rapiro_rta" {
  subnet_id      = aws_subnet.rapiro_subnet.id
  route_table_id = aws_route_table.rapiro_rt.id
}

# ---------------------------------------------------------------------------
# SECURITY GROUP — Dashboard EC2
# ---------------------------------------------------------------------------

resource "aws_security_group" "dashboard_sg" {
  name        = "rapiro-dashboard-sg-${var.environment}"
  description = "Dashboard RAPIRO: HTTP publico y SSH"
  vpc_id      = aws_vpc.rapiro_vpc.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTP dashboard"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.common_tags, { Name = "rapiro-dashboard-sg" })
}

# ---------------------------------------------------------------------------
# IAM — Rol para la EC2 (lectura de DynamoDB sin credenciales hardcodeadas)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "dashboard_ec2_role" {
  name = "rapiro-dashboard-ec2-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })

  tags = var.common_tags
}

resource "aws_iam_role_policy" "dashboard_ec2_policy" {
  name = "rapiro-dashboard-ec2-policy-${var.environment}"
  role = aws_iam_role.dashboard_ec2_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["dynamodb:Scan", "dynamodb:Query", "dynamodb:GetItem"]
        Resource = aws_dynamodb_table.sessions.arn
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject", "s3:GetObject", "s3:ListBucket", "s3:DeleteObject"]
        Resource = [
          aws_s3_bucket.documents.arn,
          "${aws_s3_bucket.documents.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_instance_profile" "dashboard_profile" {
  name = "rapiro-dashboard-profile-${var.environment}"
  role = aws_iam_role.dashboard_ec2_role.name
}

# ---------------------------------------------------------------------------
# PAR DE CLAVES SSH
# ---------------------------------------------------------------------------

resource "aws_key_pair" "rapiro_key" {
  key_name   = "rapiro-key-${var.environment}"
  public_key = var.ec2_public_key
  tags       = var.common_tags
}

# ---------------------------------------------------------------------------
# EC2 — Dashboard web (FastAPI + Docker)
# ---------------------------------------------------------------------------

resource "aws_instance" "dashboard" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = "t3.small"
  subnet_id              = aws_subnet.rapiro_subnet.id
  vpc_security_group_ids = [aws_security_group.dashboard_sg.id]
  key_name               = aws_key_pair.rapiro_key.key_name
  iam_instance_profile   = aws_iam_instance_profile.dashboard_profile.name

  root_block_device {
    volume_size = 10
    volume_type = "gp3"
  }

  user_data = <<EOF
#!/bin/bash
exec > /var/log/rapiro-dashboard-install.log 2>&1
set -e

apt-get update -y
apt-get upgrade -y
apt-get install -y docker.io docker-compose

systemctl start docker
systemctl enable docker

fallocate -l 512M /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

mkdir -p /opt/rapiro-dashboard

# ── requirements ─────────────────────────────────────────────────────────────
cat > /opt/rapiro-dashboard/requirements.txt <<'REQS'
fastapi==0.111.0
uvicorn[standard]==0.29.0
boto3==1.34.0
REQS

# ── Dockerfile ───────────────────────────────────────────────────────────────
cat > /opt/rapiro-dashboard/Dockerfile <<'DOCKERFILE'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
DOCKERFILE

# ── FastAPI app ───────────────────────────────────────────────────────────────
cat > /opt/rapiro-dashboard/app.py <<'PYAPP'
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import boto3
from boto3.dynamodb.conditions import Attr

app = FastAPI(title="RAPIRO Dashboard")

TABLE  = os.environ.get("DYNAMODB_TABLE", "rapiro_sessions_dev")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

dynamodb = boto3.resource("dynamodb", region_name=REGION)
table    = dynamodb.Table(TABLE)

CLASS_LABELS = {0: "Estudiando", 1: "Usando celular", 2: "Puesto vacio"}

@app.get("/health")
def health():
    return {"status": "ok", "table": TABLE}

@app.get("/api/sessions")
def get_sessions():
    resp  = table.scan(Limit=50)
    items = sorted(resp.get("Items", []), key=lambda x: x.get("timestamp", ""), reverse=True)
    return {"sessions": items, "count": len(items)}

@app.get("/api/stats")
def get_stats():
    resp  = table.scan()
    items = resp.get("Items", [])
    counts = {0: 0, 1: 0, 2: 0}
    for item in items:
        c = item.get("clase_detectada")
        if c is not None:
            counts[int(c)] = counts.get(int(c), 0) + 1
    total = sum(counts.values())
    pcts = {CLASS_LABELS[k]: round(v / total * 100, 1) if total else 0 for k, v in counts.items()}
    return {"total_events": total, "by_class": {CLASS_LABELS[k]: v for k, v in counts.items()}, "percentages": pcts}

HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAPIRO Guardian</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}
header{background:#1e293b;padding:1.5rem 2rem;border-bottom:1px solid #334155}
header h1{font-size:1.5rem;color:#38bdf8}
header p{color:#94a3b8;font-size:.875rem;margin-top:.25rem}
main{max-width:1100px;margin:2rem auto;padding:0 1rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;margin-bottom:1.5rem}
.card{background:#1e293b;border-radius:.75rem;padding:1.5rem;border:1px solid #334155}
.card .lbl{font-size:.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}
.card .val{font-size:2rem;font-weight:700;margin-top:.4rem}
.c0 .val{color:#22c55e}.c1 .val{color:#f59e0b}.c2 .val{color:#ef4444}.ct .val{color:#38bdf8}
.sec{background:#1e293b;border-radius:.75rem;border:1px solid #334155;margin-bottom:1.5rem}
.sec-h{padding:.75rem 1.5rem;border-bottom:1px solid #334155;font-weight:600;color:#cbd5e1}
table{width:100%;border-collapse:collapse}
th{padding:.6rem 1.5rem;text-align:left;font-size:.7rem;color:#64748b;text-transform:uppercase}
td{padding:.6rem 1.5rem;border-top:1px solid #1e293b;font-size:.85rem}
tr:hover td{background:#0f172a}
.badge{display:inline-block;padding:.15rem .55rem;border-radius:9999px;font-size:.7rem;font-weight:600}
.b0{background:#14532d;color:#86efac}.b1{background:#78350f;color:#fcd34d}.b2{background:#7f1d1d;color:#fca5a5}
.bar-row{display:flex;align-items:center;gap:1rem;padding:.6rem 1.5rem}
.bar-lbl{width:130px;font-size:.85rem;color:#cbd5e1}
.bar-track{flex:1;background:#334155;border-radius:9999px;height:8px}
.bar-fill{height:8px;border-radius:9999px;transition:width .5s}
.bar-pct{width:45px;text-align:right;font-size:.85rem;color:#94a3b8}
.foot{text-align:center;color:#475569;font-size:.75rem;padding:1rem}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#22c55e;margin-right:.4rem;animation:p 2s infinite}
@keyframes p{0%,100%%{opacity:1}50%%{opacity:.4}}
</style>
</head>
<body>
<header>
  <h1>RAPIRO Guardian — Dashboard</h1>
  <p><span class="dot"></span>Monitoreo de sesiones de estudio en tiempo real</p>
</header>
<main>
  <div class="cards">
    <div class="card ct"><div class="lbl">Total eventos</div><div class="val" id="total">—</div></div>
    <div class="card c0"><div class="lbl">Estudiando</div><div class="val" id="p0">—</div></div>
    <div class="card c1"><div class="lbl">Usando celular</div><div class="val" id="p1">—</div></div>
    <div class="card c2"><div class="lbl">Puesto vacio</div><div class="val" id="p2">—</div></div>
  </div>
  <div class="sec">
    <div class="sec-h">Distribucion por clase</div>
    <div id="bars"></div>
  </div>
  <div class="sec">
    <div class="sec-h">Ultimos eventos</div>
    <table>
      <thead><tr><th>Timestamp</th><th>Sesion</th><th>Estado</th><th>Confianza</th><th>Latencia</th></tr></thead>
      <tbody id="tbody"><tr><td colspan="5" style="text-align:center;color:#475569;padding:2rem">Cargando...</td></tr></tbody>
    </table>
  </div>
  <div class="foot" id="foot">Actualizando cada 10 segundos</div>
</main>
<script>
const LABELS=["Estudiando","Usando celular","Puesto vacio"];
const COLORS=["#22c55e","#f59e0b","#ef4444"];
const BADGES=["b0","b1","b2"];
async function loadStats(){
  try{
    const d=await(await fetch("/api/stats")).json();
    document.getElementById("total").textContent=d.total_events;
    const p=d.percentages;
    document.getElementById("p0").textContent=(p["Estudiando"]||0)+"%";
    document.getElementById("p1").textContent=(p["Usando celular"]||0)+"%";
    document.getElementById("p2").textContent=(p["Puesto vacio"]||0)+"%";
    document.getElementById("bars").innerHTML=LABELS.map((l,i)=>{
      const v=p[l]||0;
      return '<div class="bar-row"><div class="bar-lbl">'+l+'</div><div class="bar-track"><div class="bar-fill" style="width:'+v+'%;background:'+COLORS[i]+'"></div></div><div class="bar-pct">'+v+'%</div></div>';
    }).join("");
  }catch(e){}
}
async function loadSessions(){
  try{
    const d=await(await fetch("/api/sessions")).json();
    if(!d.sessions||!d.sessions.length){
      document.getElementById("tbody").innerHTML='<tr><td colspan="5" style="text-align:center;color:#475569;padding:2rem">Sin eventos aun</td></tr>';
      return;
    }
    document.getElementById("tbody").innerHTML=d.sessions.slice(0,20).map(s=>{
      const c=parseInt(s.clase_detectada);
      const ts=(s.timestamp||"—").replace("T"," ").substring(0,19);
      const conf=s.confianza?(parseFloat(s.confianza)*100).toFixed(1)+"%":"—";
      const lat=s.latencia_ms?s.latencia_ms+" ms":"—";
      const sid=(s.session_id||"—").substring(0,8);
      return '<tr><td>'+ts+'</td><td style="font-family:monospace;font-size:.75rem">'+sid+'</td><td><span class="badge '+BADGES[c]+'">'+LABELS[c]+'</span></td><td>'+conf+'</td><td>'+lat+'</td></tr>';
    }).join("");
    document.getElementById("foot").textContent="Ultima actualizacion: "+new Date().toLocaleTimeString("es-AR");
  }catch(e){}
}
function refresh(){loadStats();loadSessions();}
refresh();
setInterval(refresh,10000);
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTML
PYAPP

# ── docker-compose ────────────────────────────────────────────────────────────
cat > /opt/rapiro-dashboard/docker-compose.yml <<COMPOSE
services:
  dashboard:
    build: .
    ports:
      - "80:8000"
    environment:
      - AWS_DEFAULT_REGION=${var.aws_region}
      - DYNAMODB_TABLE=${aws_dynamodb_table.sessions.name}
    restart: unless-stopped
COMPOSE

cd /opt/rapiro-dashboard
docker-compose up -d --build

echo "=== Dashboard RAPIRO instalado correctamente ===" >> /var/log/rapiro-dashboard-install.log
EOF

  tags = merge(var.common_tags, { Name = "rapiro-dashboard-${var.environment}" })
}

# ---------------------------------------------------------------------------
# ELASTIC IP — IP fija para el dashboard
# ---------------------------------------------------------------------------

resource "aws_eip" "dashboard_eip" {
  instance = aws_instance.dashboard.id
  domain   = "vpc"
  tags     = merge(var.common_tags, { Name = "rapiro-dashboard-eip" })
}

# ---------------------------------------------------------------------------
# S3 — Modelo TFLite
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "model" {
  bucket = "rapiro-model-${var.environment}-${data.aws_caller_identity.current.account_id}"
  tags   = var.common_tags
}

resource "aws_s3_bucket_versioning" "model" {
  bucket = aws_s3_bucket.model.id
  versioning_configuration {
    status = "Enabled"
  }
}

# ---------------------------------------------------------------------------
# IAM — Rol para Lambda de inferencia
# ---------------------------------------------------------------------------

resource "aws_iam_role" "inference_role" {
  name = "rapiro-inference-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })

  tags = var.common_tags
}

resource "aws_iam_role_policy" "inference_policy" {
  name = "rapiro-inference-policy-${var.environment}"
  role = aws_iam_role.inference_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "${aws_s3_bucket.model.arn}/*"
      },
      {
        Effect   = "Allow"
        Action   = ["polly:SynthesizeSpeech"]
        Resource = "*"
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
# Lambda — Inferencia de imágenes (zip con inference_handler.py)
# Soporta TFLite cuando se agrega el Layer; funciona como endpoint HTTP
# ---------------------------------------------------------------------------

resource "aws_lambda_function" "inference" {
  filename         = "../lambda/inference.zip"
  function_name    = "rapiro-inference-${var.environment}"
  role             = aws_iam_role.inference_role.arn
  handler          = "inference_handler.lambda_handler"
  runtime          = "python3.11"
  source_code_hash = filebase64sha256("../lambda/inference.zip")
  timeout          = 30
  memory_size      = 512

  environment {
    variables = {
      MODEL_BUCKET = aws_s3_bucket.model.bucket
      MODEL_KEY    = "model.tflite"
    }
  }

  tags = var.common_tags
}

resource "aws_lambda_function_url" "inference" {
  function_name      = aws_lambda_function.inference.function_name
  authorization_type = "AWS_IAM"
}

resource "aws_cloudwatch_log_group" "inference" {
  name              = "/aws/lambda/${aws_lambda_function.inference.function_name}"
  retention_in_days = 7
  tags              = var.common_tags
}

# ---------------------------------------------------------------------------
# IAM — Usuario dedicado para Raspberry Pi
# Credenciales para boto3: IoT, Lambda invoke, Polly, S3 documentos
# ---------------------------------------------------------------------------

resource "aws_iam_user" "rapiro_pi" {
  name = "rapiro-pi-${var.environment}"
  tags = var.common_tags
}

resource "aws_iam_access_key" "rapiro_pi" {
  user = aws_iam_user.rapiro_pi.name
}

resource "aws_iam_user_policy" "rapiro_pi" {
  name = "rapiro-pi-policy-${var.environment}"
  user = aws_iam_user.rapiro_pi.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["lambda:InvokeFunction", "lambda:InvokeFunctionUrl"]
        Resource = aws_lambda_function.inference.arn
      },
      {
        Effect   = "Allow"
        Action   = ["iot:Publish", "iot:Connect"]
        Resource = "arn:aws:iot:${var.aws_region}:*:topic/${var.iot_topic}"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject", "s3:GetObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.documents.arn,
          "${aws_s3_bucket.documents.arn}/*",
          aws_s3_bucket.model.arn,
          "${aws_s3_bucket.model.arn}/*"
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["polly:SynthesizeSpeech"]
        Resource = "*"
      }
    ]
  })
}
