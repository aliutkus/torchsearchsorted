#include <iostream>
#include <climits>
#include <assert.h>

__device__  __host__
int midpoint(int a, int b)
{
    return a + (b-a)/2;
}

__device__ __host__
int eval(int A[], int i, int val, int imin, int imax)
{

    int low = (A[i] <= val);
    int high = (A[i+1] > val);

    if (low && high) {
        return 0;
    } else if (low) {
        return -1;
    } else {
        return 1;
    }
}

__device__ __host__
int binary_search(int A[], int val, int imin, int imax)
{
    while (imax >= imin) {
        int imid = midpoint(imin, imax);
        int e = eval(A, imid, val, imin, imax);
        if(e == 0) {
            return imid;
        } else if (e < 0) {
            imin = imid;
        } else {
            imax = imid;
        }
    }

    return -1;
}


__device__ __host__
int linear_search(int A[], int val, int imin, int imax)
{
    int res = -1;
    for(int i=imin; i<(imax-1); i++) {
        if (A[i+1] > val) {
            res = i;
            break;
        }
    }

    return res;
}

template<int version>
__global__
void search(int * source, int * result, int Nin, int Nout)
{
    extern __shared__ int buff[];
    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    int val = INT_MAX;
    if (tid < Nin) val = source[threadIdx.x];
    buff[threadIdx.x] = val;
    __syncthreads();

    int res;
    switch(version) {

        case 0:
        res = binary_search(buff, threadIdx.x, 0, blockDim.x);
        break;

        case 1:
        res = linear_search(buff, threadIdx.x, 0, blockDim.x);
        break;
    }

    if (tid < Nout) result[tid] = res;
}

int main(void)
{
    const int inputLength = 128000;
    const int isize = inputLength * sizeof(int);
    const int outputLength = 256;
    const int osize = outputLength * sizeof(int);

    int * hostInput = new int[inputLength];
    int * hostOutput = new int[outputLength];
    int * deviceInput;
    int * deviceOutput;

    for(int i=0; i<inputLength; i++) {
        hostInput[i] = -200 + 5*i;
    }

    cudaMalloc((void**)&deviceInput, isize);
    cudaMalloc((void**)&deviceOutput, osize);

    cudaMemcpy(deviceInput, hostInput, isize, cudaMemcpyHostToDevice);

    dim3 DimBlock(256, 1, 1);
    dim3 DimGrid(1, 1, 1);
    DimGrid.x = (outputLength / DimBlock.x) +
                ((outputLength % DimBlock.x > 0) ? 1 : 0);
    size_t shmsz = DimBlock.x * sizeof(int);

    for(int i=0; i<5; i++) {
        search<1><<<DimGrid, DimBlock, shmsz>>>(deviceInput, deviceOutput,
                inputLength, outputLength);
    }

    for(int i=0; i<5; i++) {
        search<0><<<DimGrid, DimBlock, shmsz>>>(deviceInput, deviceOutput,
                inputLength, outputLength);
    }

    cudaMemcpy(hostOutput, deviceOutput, osize, cudaMemcpyDeviceToHost);

    for(int i=0; i<outputLength; i++) {
        int idx = hostOutput[i];
        int tidx = i % DimBlock.x;
        assert( (hostInput[idx] <= tidx) && (tidx < hostInput[idx+1]) );
    }
    cudaDeviceReset();

    return 0;
}
