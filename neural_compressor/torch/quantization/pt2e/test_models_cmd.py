# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

# Suppose you're running an external script, use subprocess to handle it
import subprocess

import pandas as pd

# Define the models to be quantized
models = [
    "/mnt/disk4/modelHub/gpt-j-6b",
    "/mnt/disk4/modelHub/llama-2-7b-chat-hg",
    "/mnt/disk4/modelHub/Mistral-7B-v0.1",
    "gpt2",
    "facebook/opt-125m",
]

# Initialize a list to store the results
results = []


# Function to run the quantization process
def quantize_model(model_name_or_path):
    try:
        from x86_quantizer_hf import quant

        # Run the quantization process
        quant_status = quant(model_name_or_path=model_name_or_path)
        if quant_status:
            return "Success"
        else:
            return "Quantization finished, but output is None"
    except subprocess.CalledProcessError as e:
        # Return error message or a custom message on failure
        return f"Failed: {str(e)}"


# Loop through the models, attempt to quantize them, and store the results
for model_path in models:
    status = quantize_model(model_path)
    results.append({"Model Path": model_path, "Quantization Status": status})

# Convert the results to a DataFrame
df = pd.DataFrame(results)

# Save the DataFrame to an Excel file
df.to_excel("quantization_results.xlsx", index=False)
