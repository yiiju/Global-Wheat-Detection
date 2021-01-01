python3 test.py --weights ./runs/exp64_yolov4-csp/weights/last_yolov4-csp.pt \
	        --data data/wheat.yaml \
		--img-size 1024 \
		--cfg models/yolov4-csp_1024.cfg \
		--batch-size 30 \
		--augment \
		--device 6 \
		--names yolov4-csp \
		--task val \
		--conf 0.3