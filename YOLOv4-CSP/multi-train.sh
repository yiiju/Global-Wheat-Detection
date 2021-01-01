python3 -m torch.distributed.launch \
	--nproc_per_node 3 \
	train.py --device 7,8,9 \
	--img-size 1024 \
	--data data/wheat.yaml \
	--cfg models/yolov4-csp_1024.cfg \
	--weights 'yolov4-csp.weights' \
	--name yolov4-csp --batch-size 12 \
	--notest \
	--single-cls \
	--cache-images