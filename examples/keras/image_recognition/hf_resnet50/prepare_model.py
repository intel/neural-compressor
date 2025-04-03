import tensorflow as tf
from transformers import TFResNetForImageClassification

# Download Resnet50 from HuggingFace and save it as saved model
# It will be saved at resnet50-saved-model/saved_model/1
model = TFResNetForImageClassification.from_pretrained("microsoft/resnet-50")
model.save_pretrained('resnet50-saved-model', saved_model=True)
