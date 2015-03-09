CC=gcc
CFLAGS=-c -std=c99 -Wall
LDFLAGS= -lOpenCL -lm
SOURCES= main.c opencl_error.c opencl_handler.c opencl_test.c lodepng.c gauss_kernel.c opencl_util.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=hh_freak_detector
VPATH=src/

all: $(SOURCES) $(EXECUTABLE)

debug: CFLAGS += -DDEBUG -g
debug: all

profile: CFLAGS += -DPROFILE
profile: all

clean: 
	rm $(OBJECTS) $(EXECUTABLE)
	    
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

.c.o:
	$(CC) $(CFLAGS) $< -o $@

