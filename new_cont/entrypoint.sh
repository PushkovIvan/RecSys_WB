#!/bin/bash

if [ "$1" = "1" ]; then
    python /app/fit.py
elif [ "$1" = "2" ]; then
    python /app/fitten.py
else
    echo "Некорректный выбор"
fi