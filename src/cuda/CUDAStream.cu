
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code
#include "CUDAStream.h"


#define PAD_EVERY_X_ELEMENTS (16 * (8 / sizeof(T))) 
#define CONFLICT_FREE_OFFSET(n) ((n) / PAD_EVERY_X_ELEMENTS)   

void check_error(void)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(err);
  }
}


template <class T>
CUDAStream<T>::CUDAStream(const int ARRAY_SIZE, const int device_index)
{

  // The array size must be divisible by TBSIZE for kernel launches
  if (ARRAY_SIZE % TBSIZE != 0)
  {
    std::stringstream ss;
    ss << "Array size must be a multiple of " << TBSIZE;
    throw std::runtime_error(ss.str());
  }

  // Set device
  int count;
  cudaGetDeviceCount(&count);
  check_error();
  if (device_index >= count)
    throw std::runtime_error("Invalid device index");
  cudaSetDevice(device_index);
  check_error();

  // Print out device information
  std::cout << "Using CUDA device " << getDeviceName(device_index) << std::endl;
  std::cout << "Hello from Alex!" << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;
#if defined(MANAGED)
  std::cout << "Memory: MANAGED" << std::endl;
#elif defined(PAGEFAULT)
  std::cout << "Memory: PAGEFAULT" << std::endl;
#else
  std::cout << "Memory: DEFAULT" << std::endl;
#endif
  array_size = ARRAY_SIZE;

  // Query device for sensible dot kernel block count
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device_index);
  check_error();
  dot_num_blocks = props.multiProcessorCount * 4;

  // Allocate the host array for partial sums for dot kernels
  sums = (T*)malloc(sizeof(T) * dot_num_blocks);

  // Calculates length of the array needed to carry out scan - we need additional space to deal with intermediate arrays, which are concatenated to the end of the overall array (d) and accessed via pointer arithmetic
  scan_array_size = (int)(array_size * (1.0f - powf(1.0f / (TBSIZE*2), ceilf(logf(array_size) / logf((float)(TBSIZE*2))))) / (1.0f - 1.0f / (TBSIZE*2)));   

  size_t array_bytes = sizeof(T);
  array_bytes *= ARRAY_SIZE;
  size_t array_bytes_d = scan_array_size * sizeof(T);
  size_t total_bytes = array_bytes * 4;
  std::cout << "Reduction kernel config: " << dot_num_blocks << " groups of (fixed) size " << TBSIZE << std::endl;

  // Check buffers fit on the device
  if (props.totalGlobalMem < total_bytes)
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");

  // Create device buffers
#if defined(MANAGED)
  cudaMallocManaged(&d_a, array_bytes);
  check_error();
  cudaMallocManaged(&d_b, array_bytes);
  check_error();
  cudaMallocManaged(&d_c, array_bytes);
  check_error();
  cudaMallocManaged(&d_d, array_bytes_d);
  check_error();
  cudaMallocManaged(&d_sum, dot_num_blocks*sizeof(T));
  check_error();
#elif defined(PAGEFAULT)
  d_a = (T*)malloc(array_bytes);
  d_b = (T*)malloc(array_bytes);
  d_c = (T*)malloc(array_bytes);
  d_d = (T*)malloc(array_bytes_d);
  d_sum = (T*)malloc(sizeof(T)*dot_num_blocks);
#else
  cudaMalloc(&d_a, array_bytes);
  check_error();
  cudaMalloc(&d_b, array_bytes);
  check_error();
  cudaMalloc(&d_c, array_bytes);
  check_error();
  cudaMalloc(&d_d, array_bytes_d);
  check_error();
  cudaMalloc(&d_sum, dot_num_blocks*sizeof(T));
  check_error();
#endif
}


template <class T>
CUDAStream<T>::~CUDAStream()
{
  free(sums);

#if defined(PAGEFAULT)
  free(d_a);
  free(d_b);
  free(d_c);
  free(d_d);
  free(d_sum);
#else
  cudaFree(d_a);
  check_error();
  cudaFree(d_b);
  check_error();
  cudaFree(d_c);
  check_error();
  cudaFree(d_d);
  check_error();
  cudaFree(d_sum);
  check_error();
#endif
}


template <typename T>
__global__ void init_kernel(T * a, T * b, T * c, T initA, T initB, T initC)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = initA;
  b[i] = initB;
  c[i] = initC;  
}

template <typename T>
__global__ void init_kernel_just_d(T * d)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  d[i] = i+1;
}


template <class T>
void CUDAStream<T>::init_arrays(T initA, T initB, T initC)
{
  init_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c, initA, initB, initC);
  check_error();
  init_kernel_just_d<<<scan_array_size/TBSIZE, TBSIZE>>>(d_d);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <class T>
void CUDAStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, std::vector<T>& d)
{
  
// Copy device memory to host
#if defined(PAGEFAULT) || defined(MANAGED)
  cudaDeviceSynchronize();
  for (int i = 0; i < array_size; i++)
  {
    a[i] = d_a[i];
    b[i] = d_b[i];
    c[i] = d_c[i];
  }

  for (int i = 0; i < scan_array_size; i++)
  {
    d[i] = d_d[i];
  }

#else
  cudaMemcpy(a.data(), d_a, a.size()*sizeof(T), cudaMemcpyDeviceToHost);
  check_error();
  cudaMemcpy(b.data(), d_b, b.size()*sizeof(T), cudaMemcpyDeviceToHost);
  check_error();
  cudaMemcpy(c.data(), d_c, c.size()*sizeof(T), cudaMemcpyDeviceToHost);
  check_error();
  cudaMemcpy(d.data(), d_d, d.size()*sizeof(T), cudaMemcpyDeviceToHost);
  check_error();
#endif
}


template <typename T>
__global__ void copy_kernel(const T * a, T * c)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i];
}

template <class T>
void CUDAStream<T>::copy()
{
  copy_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void mul_kernel(T * b, const T * c)
{
  const T scalar = startScalar;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  b[i] = scalar * c[i];
}

template <class T>
void CUDAStream<T>::mul()
{
  mul_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_b, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void add_kernel(const T * a, const T * b, T * c)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i] + b[i];
}

template <class T>
void CUDAStream<T>::add()
{
  add_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void triad_kernel(T * a, const T * b, const T * c)
{
  const T scalar = startScalar;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = b[i] + scalar * c[i];
}

template <class T>
void CUDAStream<T>::triad()
{
  triad_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void nstream_kernel(T * a, const T * b, const T * c)
{
  const T scalar = startScalar;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] += b[i] + scalar * c[i];
}

template <class T>
void CUDAStream<T>::nstream()
{
  nstream_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <class T>
__global__ void dot_kernel(const T * a, const T * b, T * sum, int array_size)
{
  __shared__ T tb_sum[TBSIZE];

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t local_i = threadIdx.x;

  tb_sum[local_i] = {};
  for (; i < array_size; i += blockDim.x*gridDim.x)
    tb_sum[local_i] += a[i] * b[i];

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
  {
    __syncthreads();
    if (local_i < offset)
    {
      tb_sum[local_i] += tb_sum[local_i+offset];
    }
  }

  if (local_i == 0)
    sum[blockIdx.x] = tb_sum[local_i];
}

template <class T>
T CUDAStream<T>::dot()
{
  dot_kernel<<<dot_num_blocks, TBSIZE>>>(d_a, d_b, d_sum, array_size);
  check_error();

#if defined(MANAGED) || defined(PAGEFAULT)
  cudaDeviceSynchronize();
  check_error();
#else
  cudaMemcpy(sums, d_sum, dot_num_blocks*sizeof(T), cudaMemcpyDeviceToHost);
  check_error();
#endif

  T sum = 0.0;
  for (int i = 0; i < dot_num_blocks; i++)
  {
#if defined(MANAGED) || defined(PAGEFAULT)
    sum += d_sum[i];
#else
    sum += sums[i];
#endif
  }

  return sum;
}

template <typename T>
__global__ void scan_kernel(T * g_idata, int n, T * sums)
{
    //Holds the part that we are working on
    extern __shared__ unsigned char shared_mem[];
    T* temp = reinterpret_cast<T*>(shared_mem);

    //Positioning, thidRaw is just the raw thread number wheras thid also takes into account the current block
    int blockStart = blockIdx.x * (n / (gridDim.x));  
    int thidRaw = threadIdx.x;
    int thid = blockStart + thidRaw;
    
    //n is total size of array whereas the value n is the size of the sub-array the current block is dealing with 
    int localN = n / gridDim.x;  

    int offset = 1;

    //Loads two (arbitrary) elements that we know should not cause a bank conflict (e1 - element1)
    int e1LocalPos = thidRaw;
    int e2LocalPos = thidRaw + (localN/2);
    int e1GlobalPos = thid;
    int e2GlobalPos = thid + (localN/2); 

    //Load data into shared memory, add bank conflict avoiding offset
    int bankOffsetA = CONFLICT_FREE_OFFSET(e1LocalPos);
    int bankOffsetB = CONFLICT_FREE_OFFSET(e2LocalPos);

   
    if (e1GlobalPos < n) temp[e1LocalPos + bankOffsetA] = g_idata[e1GlobalPos];
    if (e2GlobalPos < n) temp[e2LocalPos + bankOffsetB] = g_idata[e2GlobalPos];


    //Build sum in-place, up the tree
    for (int d = localN >> 1; d > 0; d >>= 1){ 
      __syncthreads();
      if (thidRaw < d){ //thid is half of what it should be hence d essentially n here 
        int ai = offset * (2 * thidRaw + 1) - 1;
        int bi = offset * (2 * thidRaw + 2) - 1;
        ai += CONFLICT_FREE_OFFSET(ai);
        bi += CONFLICT_FREE_OFFSET(bi);
        temp[bi] += temp[ai];
      }
      offset *= 2;
    }

    //When we reach end of the first phase, we save the value of the total sum in the array sums
    if (thidRaw == 0){
        int last = localN - 1 + CONFLICT_FREE_OFFSET(localN - 1);
        sums[blockIdx.x] = temp[last];
        temp[last] = 0.0;
    }

    //Traverse down tree & build scan
    for (int d = 1; d < localN; d *= 2) {  
        offset >>= 1;
        __syncthreads();
        if (thidRaw < d){
            int ai = offset * (2 * thidRaw + 1) - 1;
            int bi = offset * (2 * thidRaw + 2) - 1;

            int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
            int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

            T t = temp[ai + bankOffsetA];

            temp[ai + bankOffsetA] = temp[bi + bankOffsetB];
            temp[bi + bankOffsetB] += t;
        }
    }    

    __syncthreads();

  if (e1GlobalPos < n) g_idata[e1GlobalPos] = temp[e1LocalPos + bankOffsetA];
  if (e2GlobalPos < n) g_idata[e2GlobalPos] = temp[e2LocalPos + bankOffsetB];

}

template <typename T>
__global__ void addblockSums(T * g_idata, T * icnr, int blockSize)
{
  __shared__ T icnrCurrent;

    int thid = threadIdx.x;
    int bid = blockIdx.x;

    if (thid == 0) {icnrCurrent = icnr[bid];}

    __syncthreads();  // Ensure all threads see the loaded shared value

    int indexA = bid * blockSize + thid;
    int indexB = indexA + (blockSize / 2);

    g_idata[indexA] += icnrCurrent;
    g_idata[indexB] += icnrCurrent;
}


//Recursive algorithm that calls itself when the number of blocks (hence length of intermediate array) exceeds TBSIZE elements
template <class T>
void recursiveScan(T * d_input, int current_length){
    int B = TBSIZE * 2;

    //We then round up (to the nearest power of two) if we find our intermediate array is not a power of two
    current_length = ((current_length & (current_length - 1)) != 0) ? 1 << (int)ceil(log2(current_length)) : current_length;
    int num_of_blocks = (current_length/B > 0) ? current_length/B : 1;

    T* sums = &d_input[current_length]; 

    int shared_size = (B + CONFLICT_FREE_OFFSET(B)) * sizeof(T);
    // printf("\n\n B: %d, PAD_EVERY_X_ELEMENTS: %d, T: %d, shared_size: %d, current_length: %d\n\n", B, PAD_EVERY_X_ELEMENTS, sizeof(T), shared_size, current_length);
    int sums_size = (num_of_blocks + CONFLICT_FREE_OFFSET(num_of_blocks)) * sizeof(T);

    scan_kernel<<<num_of_blocks, B/2, shared_size>>>(d_input, current_length, sums);  
     
    cudaDeviceSynchronize();
    check_error();

     
    //When our intermediate array contains more elements than one block can deal with, we do a full scan on that array and only ...then... do we continue
    if (num_of_blocks > B){ //Recursive case
      recursiveScan(sums, num_of_blocks);

    } else if(num_of_blocks > 1) { //Base case
      scan_kernel<<<1, num_of_blocks/2, sums_size>>>(sums, num_of_blocks, sums); 
      cudaDeviceSynchronize();
      check_error();
    }

    if (num_of_blocks > 1)
    {
      addblockSums<<<num_of_blocks, B/2>>>(d_input, sums, B);  
      cudaDeviceSynchronize();
      check_error();
    }
    
}
    
template <class T>
void CUDAStream<T>::scan()
{
    recursiveScan(d_d, array_size);

    // T* result = new T[scan_array_size];
    // cudaMemcpy(result, d_d, scan_array_size * sizeof(T), cudaMemcpyDeviceToHost);

    // printf("\nLast few prefix sums:\n");
    // for (int i = array_size - 3; i < array_size; ++i) {printf("%f ", result[i]);}
    // printf("\n");

    // delete[] result;

}

void listDevices(void)
{
  // Get number of devices
  int count;
  cudaGetDeviceCount(&count);
  check_error();

  // Print device names
  if (count == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < count; i++)
    {
      std::cout << i << ": " << getDeviceName(i) << std::endl;
    }
    std::cout << std::endl;
  }
}


std::string getDeviceName(const int device)
{
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);
  check_error();
  return std::string(props.name);
}


std::string getDeviceDriver(const int device)
{
  cudaSetDevice(device);
  check_error();
  int driver;
  cudaDriverGetVersion(&driver);
  check_error();
  return std::to_string(driver);
}

template class CUDAStream<float>;
template class CUDAStream<double>;
