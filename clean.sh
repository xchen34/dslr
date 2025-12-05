#!/bin/bash

arr=(`find ./ -maxdepth 1 -name "*.png"`)

echo "🧹 Perform super cleaning now..."
if [ ${#arr[@]} -gt 0 ]; then
    echo "🫧 Clearning *.png..."
    rm *.png
fi

arr=(`find ./ -maxdepth 1 -name "*.json"`)
if [ ${#arr[@]} -gt 0 ]; then
    echo "🧽 Clearning *.json..."
    rm *.json
fi

if [ -e 'houses.csv' ]; then
    echo "🫧 Clearning houses.csv..."
    rm houses.csv
fi

if [ -d '.venv' ]; then
    echo "🧽 Clearning python virtual env..."
    rm -r .venv
fi

echo "✨ All done ✨"