LOCAL_PATH?=.
GENERATE_PATH:=$(LOCAL_PATH)

encodekernels:
	$(GENERATE_PATH)/script/encode_kernels.sh $(GENERATE_PATH)/kernels/ > $(GENERATE_PATH)/src/_opencl_kernels.h
