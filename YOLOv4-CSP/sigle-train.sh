python3   train.py --device 5 \
	--img 1024 \
	--data data/wheat.yaml \
	--cfg models/yolov4-csp_1024.cfg \
	--weights 'yolov4-csp.weights' \
	--name yolov4-csp \
	--batch-size 3 \
	--notest \
	--single-cls \
	--cache-images