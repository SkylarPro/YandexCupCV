pip install --no-index --no-deps --target ./py_packages ./wheels/*
git clone https://github.com/sberbank-ai/ru-clip
PYTHONPATH=./py_packages:$PYTHONPATH python3 setup_cfg.py
echo "Setup finished successfully"