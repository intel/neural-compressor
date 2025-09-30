INSTALLATION_PATH=$(python3 -c "import sys; import yolov5; p=sys.modules['yolov5'].__file__; print(p.replace('/__init__.py', ''))")
python $INSTALLATION_PATH/models/tf.py --weights yolov5/yolov5s.pt
python $INSTALLATION_PATH/export.py --weights yolov5/yolov5s.pt --include pb