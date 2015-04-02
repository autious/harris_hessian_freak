CC?=gcc #Use gcc if env doesn't specify otherwise
CFLAGS=-c -std=c99 -Wall -Iinclude
LDFLAGS= -lOpenCL -lm
LIBRARY_SOURCES= opencl_error.c opencl_loader.c opencl_test.c lodepng.c gauss_kernel.c opencl_util.c opencl_program.c opencl_fd.c harris_hessian_freak.c util.c freak.c opencl_timer.c
LIBRARY_OBJECTS=$(addprefix obj/,$(LIBRARY_SOURCES:.c=.o))

PROGRAM_SOURCES= main.c
PROGRAM_OBJECTS=$(addprefix obj/,$(PROGRAM_SOURCES:.c=.o))


EXECUTABLE=harris_hessian_freak
LIBRARY=harris_hessian_freak.a
VPATH=src/

all: $(EXECUTABLE)

lib: $(LIBRARY)

debug: CFLAGS += -DDEBUG -DPROFILE -g
debug: all

profile: CFLAGS += -DPROFILE
profile: all

clean: 
	rm $(PROGRAM_OBJECTS) $(LIBRARY_OBJECTS) $(EXECUTABLE) $(LIBRARY)

$(EXECUTABLE): $(LIBRARY) $(PROGRAM_OBJECTS)
	$(CC) $(PROGRAM_OBJECTS) $(LIBRARY) $(LDFLAGS) -o $@

$(LIBRARY): $(LIBRARY_OBJECTS)
	ar -cvq $(LIBRARY) $(LIBRARY_OBJECTS)

obj/%.o: %.c
	@mkdir -p obj/
	$(CC) $(CFLAGS) $< -o $@

opencl_program.c: encodekernels

encodekernels:
	./script/encode_kernels.sh kernels/ > src/_opencl_kernels.h

install:
	cp hh_freak_detector /usr/local/bin/
