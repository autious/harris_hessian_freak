CC=gcc
CFLAGS=-c -std=c99 -Wall
LDFLAGS= -lOpenCL -lm
SOURCES= main.c opencl_error.c opencl_handler.c opencl_test.c lodepng.c gauss_kernel.c opencl_util.c opencl_program.c opencl_fd.c harris_hessian.c util.c freak.c
OBJECTS=$(addprefix obj/,$(SOURCES:.c=.o))
EXECUTABLE=hh_freak_detector
VPATH=src/

all: $(SOURCES) $(EXECUTABLE)

debug: CFLAGS += -DDEBUG -g
debug: all

profile: CFLAGS += -DPROFILE
profile: all

clean: 
	rm $(OBJECTS) $(EXECUTABLE)

obj/:
	mkdir -p obj
	    
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

obj/%.o: %.c obj/
	$(CC) $(CFLAGS) $< -o $@

