variable "aws_region" {
  description = "Región AWS donde se despliegan los recursos"
  type        = string
  default     = "sa-east-1"
}

variable "environment" {
  description = "Nombre del entorno (dev, prod)"
  type        = string
  default     = "dev"
}

variable "iot_topic" {
  description = "Topic MQTT que publica la Raspberry Pi"
  type        = string
  default     = "rapiro/events"
}

variable "alert_email" {
  description = "Email que recibe alertas SNS cuando el puesto está vacío"
  type        = string
}

variable "ec2_public_key" {
  description = "Contenido de la clave pública SSH para acceder a la EC2 del dashboard (cat ~/.ssh/id_rsa.pub)"
  type        = string
}

variable "common_tags" {
  description = "Tags aplicados a todos los recursos"
  type        = map(string)
  default = {
    Project   = "RAPIRO-Guardian"
    ManagedBy = "Terraform"
  }
}
