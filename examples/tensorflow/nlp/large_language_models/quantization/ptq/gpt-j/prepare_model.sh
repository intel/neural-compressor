pip install transformers==4.25.0
python prepare_model.py
mv ./gpt-j-6B/saved_model/1 ./
rm -r ./gpt-j-6B
mv ./1 ./gpt-j-6B
pip install transformers==4.35