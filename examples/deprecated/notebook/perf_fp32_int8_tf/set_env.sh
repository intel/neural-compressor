deactivate
ENV_NAME=env_intel_tf
rm -rf $ENV_NAME
python -m venv $ENV_NAME
source $ENV_NAME/bin/activate
pip install --upgrade pip
pip install intel-tensorflow matplotlib

echo "Created venv $ENV_NAME, activate by run:"
echo "source $ENV_NAME/bin/activate"