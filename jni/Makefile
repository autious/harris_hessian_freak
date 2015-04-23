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

debug-nprofile: CFLAGS += -DDEBUG -g
debug-nprofile: LDFLAGS += -lrt
debug-nprofile: all

debug-half: CFLAGS += -DHH_USE_HALF
debug-half: debug 

profile: CFLAGS += -DPROFILE
profile: LDFLAGS += -lrt
profile: all

profile-half: CFLAGS += -DHH_USE_HALF
profile-half: profile

clean: 
	rm $(PROGRAM_OBJECTS) $(LIBRARY_OBJECTS) $(EXECUTABLE) $(LIBRARY)

$(EXECUTABLE): $(LIBRARY) $(PROGRAM_OBJECTS)
	$(CC) $(PROGRAM_OBJECTS) $(LIBRARY) $(LDFLAGS) -o $@

$(LIBRARY): $(LIBRARY_OBJECTS)
	ar -cq $(LIBRARY) $(LIBRARY_OBJECTS)

obj/%.o: src/%.c 	
	@mkdir -p obj/
	$(CC) $(CFLAGS) $< -o $@

obj/opencl_program.o: src/_ref_opencl_kernels.h src/_opt_opencl_kernels.h

include Generate.mk

src/opencl_program.c: encodekernels 

src/opencl_timer.c: encodeversion

install:
	cp harris_hessian_freak /usr/local/bin/
