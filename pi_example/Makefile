CC	= g++
CFLAGS	= -O3 -lOpenCL
PI_CALC	= pi_calc
LIB	= -I ~/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc/

all:
	$(CC) $(LIB) $(CFLAGS) -o $(PI_CALC) $(PI_CALC).cpp

clean:
	rm $(PI_CALC)

test:
	./$(PI_CALC) 524288 64 

