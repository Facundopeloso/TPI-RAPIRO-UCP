#!/bin/bash
# Empaqueta el handler de Lambda en el zip que Terraform referencia.
# Ejecutar desde la raíz del proyecto: bash lambda/package.sh
set -e

cd "$(dirname "$0")"
zip event_processor.zip handler.py
echo "Lambda empaquetada: lambda/event_processor.zip"
