output "dashboard_url" {
  description = "URL pública del dashboard web (disponible ~3 min después del apply)"
  value       = "http://${aws_eip.dashboard_eip.public_ip}"
}

output "dashboard_ssh" {
  description = "Comando SSH para conectarse a la EC2 del dashboard"
  value       = "ssh -i <tu-clave-privada> ubuntu@${aws_eip.dashboard_eip.public_ip}"
}

output "dashboard_install_log" {
  description = "Comando para ver el log de instalación del dashboard"
  value       = "ssh -i <tu-clave-privada> ubuntu@${aws_eip.dashboard_eip.public_ip} 'sudo tail -50 /var/log/rapiro-dashboard-install.log'"
}

output "iot_topic" {
  description = "Topic MQTT que debe publicar la Raspberry Pi"
  value       = var.iot_topic
}

output "dynamodb_table" {
  description = "Nombre de la tabla DynamoDB donde se guardan los eventos"
  value       = aws_dynamodb_table.sessions.name
}

output "s3_bucket" {
  description = "Nombre del bucket S3 para documentos del tutor"
  value       = aws_s3_bucket.documents.bucket
}

output "sns_topic_arn" {
  description = "ARN del topic SNS para suscribir emails de alerta adicionales"
  value       = aws_sns_topic.alerts.arn
}

output "lambda_function_name" {
  description = "Nombre de la Lambda que procesa eventos de IoT Core"
  value       = aws_lambda_function.event_processor.function_name
}
