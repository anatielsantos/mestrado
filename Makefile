run-it:
	docker run -it -u $(id -u):$(id -g) --gpus all -v '$(PWD):/code' jonnison/tf-1.4-gpu
run:
	docker run -d -u $(id -u):$(id -g) --gpus all -v '$(PWD):/code' jonnison/tf-1.4-gpu python -u $(F)
