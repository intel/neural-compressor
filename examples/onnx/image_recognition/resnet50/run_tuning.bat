echo off

set input_model=%1
set config=%2
set output_model=%3
shift
shift
python main.py  --model_path %input_model% --config %config% --tune --output_model %output_model%