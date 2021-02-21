2021-02-21 02:59:08.700830: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll
2021-02-21 02:59:10.919444: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-02-21 02:59:10.919867: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library nvcuda.dll
2021-02-21 02:59:10.938674: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: GeForce RTX 2060 with Max-Q Design computeCapability: 7.5
coreClock: 1.185GHz coreCount: 30 deviceMemorySize: 6.00GiB deviceMemoryBandwidth: 245.91GiB/s
2021-02-21 02:59:10.943380: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll
2021-02-21 02:59:10.953556: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublas64_11.dll
2021-02-21 02:59:10.953588: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublasLt64_11.dll
2021-02-21 02:59:10.957183: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cufft64_10.dll
2021-02-21 02:59:10.958172: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library curand64_10.dll
2021-02-21 02:59:10.968398: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusolver64_10.dll
2021-02-21 02:59:10.971317: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusparse64_11.dll
2021-02-21 02:59:10.971975: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudnn64_8.dll
2021-02-21 02:59:10.972071: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
2021-02-21 02:59:12.125598: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-02-21 02:59:12.126570: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: GeForce RTX 2060 with Max-Q Design computeCapability: 7.5
coreClock: 1.185GHz coreCount: 30 deviceMemorySize: 6.00GiB deviceMemoryBandwidth: 245.91GiB/s
2021-02-21 02:59:12.126607: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll
2021-02-21 02:59:12.126626: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublas64_11.dll
2021-02-21 02:59:12.126641: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublasLt64_11.dll
2021-02-21 02:59:12.126653: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cufft64_10.dll
2021-02-21 02:59:12.126665: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library curand64_10.dll
2021-02-21 02:59:12.126678: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusolver64_10.dll
2021-02-21 02:59:12.126690: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusparse64_11.dll
2021-02-21 02:59:12.126702: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudnn64_8.dll
2021-02-21 02:59:12.126749: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
2021-02-21 02:59:12.652626: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-02-21 02:59:12.652707: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      0 
2021-02-21 02:59:12.652773: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1280] 0:   N 
2021-02-21 02:59:12.653004: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 4720 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2060 with Max-Q Design, pci bus id: 0000:01:00.0, compute capability: 7.5)
2021-02-21 02:59:12.653683: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-02-21 02:59:13.002722: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
2021-02-21 02:59:13.567463: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublas64_11.dll
2021-02-21 02:59:13.982665: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublasLt64_11.dll
2021-02-21 02:59:13.995581: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudnn64_8.dll
2021-02-21 02:59:14.944487: I tensorflow/core/platform/windows/subprocess.cc:308] SubProcess ended with return code: 0

2021-02-21 02:59:15.225207: I tensorflow/core/platform/windows/subprocess.cc:308] SubProcess ended with return code: 0

Traceback (most recent call last):
  File "batch_normalization.py", line 125, in <module>
    results = net.model.fit(x=generator(BATCH_SIZE_TRAIN, [trainX,trainY]), validation_data=generator(BATCH_SIZE_TEST, [testX, testY]), shuffle = True, epochs = TRAIN_EPOCHS, batch_size = BATCH_SIZE_TRAIN, validation_batch_size = BATCH_SIZE_TEST, verbose = 1, steps_per_epoch=len(trainX)/BATCH_SIZE_TRAIN, validation_steps=len(testX)/BATCH_SIZE_TEST)
  File "C:\Users\nicho\tensorflow\lib\site-packages\tensorflow\python\keras\engine\training.py", line 1100, in fit
    tmp_logs = self.train_function(iterator)
  File "C:\Users\nicho\tensorflow\lib\site-packages\tensorflow\python\eager\def_function.py", line 828, in __call__
    result = self._call(*args, **kwds)
  File "C:\Users\nicho\tensorflow\lib\site-packages\tensorflow\python\eager\def_function.py", line 855, in _call
    return self._stateless_fn(*args, **kwds)  # pylint: disable=not-callable
  File "C:\Users\nicho\tensorflow\lib\site-packages\tensorflow\python\eager\function.py", line 2942, in __call__
    return graph_function._call_flat(
  File "C:\Users\nicho\tensorflow\lib\site-packages\tensorflow\python\eager\function.py", line 1918, in _call_flat
    return self._build_call_outputs(self._inference_function.call(
  File "C:\Users\nicho\tensorflow\lib\site-packages\tensorflow\python\eager\function.py", line 555, in call
    outputs = execute.execute(
  File "C:\Users\nicho\tensorflow\lib\site-packages\tensorflow\python\eager\execute.py", line 59, in quick_execute
    tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
KeyboardInterrupt
2021-02-21 02:59:38.447327: W tensorflow/core/kernels/data/generator_dataset_op.cc:107] Error occurred when finalizing GeneratorDataset iterator: Failed precondition: Python interpreter state is not initialized. The process may be terminated.
	 [[{{node PyFunc}}]]
^C