build:
	docker build -t prednet -f Dockerfile.gpu .
	nvidia-docker run -it -v `pwd`/data:/prednet/data -v `pwd`/model_data:/prednet/model_data -v `pwd`/kitti_results:/prednet/kitti_results prednet bash
