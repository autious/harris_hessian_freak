# This is the generic makefile for building the program on a unix machine.
# The project also contains android specific makefiles to make building to andoird easier
CC?=gcc #Use gcc if env doesn't specify otherwise
CFLAGS=-c -std=c99 -Wall -Iinclude
LDFLAGS= -lOpenCL -lm

include Sources.mk
PROGRAM_SOURCES= main.c

LIBRARY_OBJECTS=$(addprefix obj/,$(LIBRARY_SOURCES:.c=.o))
PROGRAM_OBJECTS=$(addprefix obj/,$(PROGRAM_SOURCES:.c=.o))
PROGRAM_SOURCES_PATH=$(addprefix src/,$(PROGRAM_SOURCES))
LIBRARY_SOURCES_PATH=$(addprefix src/,$(LIBRARY_SOURCES))

EXECUTABLE=harris_hessian_freak
LIBRARY=harris_hessian_freak.a

all: $(EXECUTABLE)

lib: $(LIBRARY)

debug: CFLAGS += -DDEBUG -DPROFILE -g
debug: LDFLAGS += -lrt
debug: all

profile: CFLAGS += -DPROFILE
profile: LDFLAGS += -lrt
profile: all

clean: 
	rm $(PROGRAM_OBJECTS) $(LIBRARY_OBJECTS) $(EXECUTABLE) $(LIBRARY)

$(EXECUTABLE): $(LIBRARY) $(PROGRAM_OBJECTS)
	$(CC) $(PROGRAM_OBJECTS) $(LIBRARY) $(LDFLAGS) -o $@

$(LIBRARY): $(LIBRARY_OBJECTS)
	ar -cvq $(LIBRARY) $(LIBRARY_OBJECTS)

obj/%.o: src/%.c
	@mkdir -p obj/
	$(CC) $(CFLAGS) $< -o $@

include Generate.mk

src/opencl_program.c: encodekernels

src/opencl_timer.c: encodeversion

install:
	cp hh_freak_detector /usr/local/bin/
