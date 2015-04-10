LOCAL_PATH?=.
GENERATE_PATH:=$(LOCAL_PATH)

encodekernels:
	$(GENERATE_PATH)/script/encode_kernels.sh "$(GENERATE_PATH)/kernels/opt/" > "$(GENERATE_PATH)/src/_opt_opencl_kernels.h" "opt_"
	$(GENERATE_PATH)/script/encode_kernels.sh "$(GENERATE_PATH)/kernels/ref/" > "$(GENERATE_PATH)/src/_ref_opencl_kernels.h" "ref_"

encodeversion:
	$(GENERATE_PATH)/script/encode_version.sh "$(GENERATE_PATH)" > "$(GENERATE_PATH)/src/_commit_data.h"
	
