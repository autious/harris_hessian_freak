CC=gcc
CFLAGS=-c -std=c99 -Wall
LDFLAGS= -lOpenCL
SOURCES= main.c opencl_error.c opencl_handler.c opencl_test.c
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

