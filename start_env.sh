#!/bin/bash
set -e

# 1. Check if under venv
in_venv() {
    [ -n "$VIRTUAL_ENV" ]
}

# 2. Start venv if not activate
if [ ! -d .venv/ ]
then
    python3 -m venv .venv
    chmod +x .venv/bin/activate
fi

if ! in_venv; then
    if [ -f ".venv/bin/activate" ]; then
        echo "🤖 Activating python virtual environment ..."
        source .venv/bin/activate
    else
        echo "❌ Virtual environment not found... Have you create it? 🫥"
        echo "💡 Tip: Run 'python3 -m venv .venv' to create it."
        exit 1
    fi
else
    echo "📦 Already in the virtual environment..."
fi


# 3. Check and Install dependencies
echo "📚 Checking Python dependencies..."
pip install --quiet --upgrade pip setuptools wheel

if [ -f "requirements.txt" ]; then
    echo "🔍 Installing/updating packages from requirements.txt..."

    pip install -r requirements.txt -qq
else
    echo "⚠️  requirements.txt not found. Skipping package installation."
fi

echo "▶️ If you did not see (.venv) open on your terminal, run \"source .venv/bin/activate\""