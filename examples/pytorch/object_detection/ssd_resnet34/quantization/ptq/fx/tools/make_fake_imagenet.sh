#!/bin/bash

opt=-q

mkdir -p fake_imagenet/val
cd fake_imagenet/val
wget $opt https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/7weeks_old.JPG/800px-7weeks_old.JPG
wget $opt https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Welsh_Springer_Spaniel.jpg/800px-Welsh_Springer_Spaniel.jpg
wget $opt https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Jammlich_crop.jpg/800px-Jammlich_crop.jpg
wget $opt https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Pumiforme.JPG/782px-Pumiforme.JPG
wget $opt https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Sardinian_Warbler.jpg/800px-Sardinian_Warbler.jpg
wget $opt https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/Cacatua_moluccensis_-Cincinnati_Zoo-8a.jpg/512px-Cacatua_moluccensis_-Cincinnati_Zoo-8a.jpg
wget $opt https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/20180630_Tesla_Model_S_70D_2015_midnight_blue_left_front.jpg/800px-20180630_Tesla_Model_S_70D_2015_midnight_blue_left_front.jpg
wget $opt https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/Porsche_991_silver_IAA.jpg/800px-Porsche_991_silver_IAA.jpg
cd ..

cat > val_map.txt <<EOF
val/800px-Porsche_991_silver_IAA.jpg 817
val/512px-Cacatua_moluccensis_-Cincinnati_Zoo-8a.jpg 89
val/800px-Sardinian_Warbler.jpg 13
val/800px-7weeks_old.JPG 207
val/800px-20180630_Tesla_Model_S_70D_2015_midnight_blue_left_front.jpg 817
val/800px-Welsh_Springer_Spaniel.jpg 156
val/800px-Jammlich_crop.jpg 233
val/782px-Pumiforme.JPG 285
EOF

