#!/bin/bash
# Останавливает и удаляет контейнеры (логи очищаются автоматически),
# затем запускает их заново.
docker-compose down
docker-compose up -d
