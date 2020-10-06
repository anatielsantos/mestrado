run-it:
	docker run -it -u $(id -u):$(id -g) --gpus all -v '$(PWD):/code' jonnison/tf-1.4-gpu
run:
	docker run -d -u $(id -u):$(id -g) --gpus all -v '$(PWD):/code' jonnison/tf-1.4-gpu python -u $(F)
run-cpu:
	#docker run -it -u $(id -u):$(id -g) -v '$(PWD):/code' tensorflow/tensorflow bash
	docker run -it -u $(id -u):$(id -g) -v '$(PWD):/code' anatielsantos/tf-cpu bash
