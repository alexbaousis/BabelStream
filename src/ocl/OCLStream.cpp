
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "OCLStream.h"
#include <cmath>


#define PAD_EVERY_X_ELEMENTS (16 * (8 / sizeof(T))) 
#define CONFLICT_FREE_OFFSET(n) ((n) / PAD_EVERY_X_ELEMENTS)
 

// Cache list of devices
bool cached = false;
std::vector<cl::Device> devices;
void getDeviceList(void);

std::string kernels{R"CLC(

  #define PAD_EVERY_X_ELEMENTS (16 * (8 / sizeof(TYPE))) 
  #define CONFLICT_FREE_OFFSET(n) ((n) / PAD_EVERY_X_ELEMENTS)

  constant TYPE scalar = startScalar;

  kernel void init(
    global TYPE * restrict a,
    global TYPE * restrict b,
    global TYPE * restrict c,
    global TYPE * restrict d,
    TYPE initA, TYPE initB, TYPE initC)
  {
    const size_t i = get_global_id(0);
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
    d[i] = i+1;
  }

  kernel void copy(
    global const TYPE * restrict a,
    global TYPE * restrict c)
  {
    const size_t i = get_global_id(0);
    c[i] = a[i];
  }

  kernel void mul(
    global TYPE * restrict b,
    global const TYPE * restrict c)
  {
    const size_t i = get_global_id(0);
    b[i] = scalar * c[i];
  }

  kernel void add(
    global const TYPE * restrict a,
    global const TYPE * restrict b,
    global TYPE * restrict c)
  {
    const size_t i = get_global_id(0);
    c[i] = a[i] + b[i];
  }

  kernel void triad(
    global TYPE * restrict a,
    global const TYPE * restrict b,
    global const TYPE * restrict c)
  {
    const size_t i = get_global_id(0);
    a[i] = b[i] + scalar * c[i];
  }
  kernel void nstream(
    global TYPE * restrict a,
    global const TYPE * restrict b,
    global const TYPE * restrict c)
  {
    const size_t i = get_global_id(0);
    a[i] += b[i] + scalar * c[i];
  }

  kernel void stream_dot(
    global const TYPE * restrict a,
    global const TYPE * restrict b,
    global TYPE * restrict sum,
    local TYPE * restrict wg_sum,
    int array_size)
  {
    size_t i = get_global_id(0);
    const size_t local_i = get_local_id(0);
    wg_sum[local_i] = 0.0;
    for (; i < array_size; i += get_global_size(0))
      wg_sum[local_i] += a[i] * b[i];

    for (int offset = get_local_size(0) / 2; offset > 0; offset /= 2)
    {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (local_i < offset)
      {
        wg_sum[local_i] += wg_sum[local_i+offset];
      }
    }

    if (local_i == 0)
      sum[get_group_id(0)] = wg_sum[local_i];
  }

  kernel void scan(__global TYPE* g_idata, int n, __global TYPE* sums, __local TYPE* temp) {

    int group_id = get_group_id(0);
    int local_id = get_local_id(0);
    int num_groups = get_num_groups(0);

    int blockStart = group_id * (n / num_groups);
    int thidRaw = local_id;
    int thid = blockStart + thidRaw;

    int localN = n / num_groups;
    int offset = 1;

    // Positions of elements in local and global memory
    int e1LocalPos = thidRaw;
    int e2LocalPos = thidRaw + (localN / 2);
    int e1GlobalPos = thid;
    int e2GlobalPos = thid + (localN / 2);

    int bankOffsetA = CONFLICT_FREE_OFFSET(e1LocalPos);
    int bankOffsetB = CONFLICT_FREE_OFFSET(e2LocalPos);

    // Load data into local memory with padding
    if (e1GlobalPos < n) { temp[e1LocalPos + bankOffsetA] = g_idata[e1GlobalPos];} 
    if (e2GlobalPos < n) { temp[e2LocalPos + bankOffsetB] = g_idata[e2GlobalPos];} 

    barrier(CLK_LOCAL_MEM_FENCE);

    // Up-sweep / reduce phase
    for (int d = localN >> 1; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (thidRaw < d) {
            int ai = offset * (2 * thidRaw + 1) - 1;
            int bi = offset * (2 * thidRaw + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Save total sum of this block and clear last element for downsweep
    if (thidRaw == 0) {
        int last = localN - 1 + CONFLICT_FREE_OFFSET(localN - 1);
        sums[group_id] = temp[last];
        temp[last] = 0.0f;
    }

    // Down-sweep phase
    for (int d = 1; d < localN; d <<= 1) {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (thidRaw < d) {
            int ai = offset * (2 * thidRaw + 1) - 1;
            int bi = offset * (2 * thidRaw + 2) - 1;
            int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
            int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

            TYPE t = temp[ai + bankOffsetA]; 
            temp[ai + bankOffsetA] = temp[bi + bankOffsetB];
            temp[bi + bankOffsetB] += t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);


    // Write results back to global memory
    if (e1GlobalPos < n) {
        g_idata[e1GlobalPos] = temp[e1LocalPos + bankOffsetA];
    }
    if (e2GlobalPos < n) {
        g_idata[e2GlobalPos] = temp[e2LocalPos + bankOffsetB];
    }
    
  }

  kernel void addblockSums(__global TYPE* g_idata, __global TYPE* icnr, int blockSize) {
    __local TYPE icnrCurrent;

    int thid = get_local_id(0);
    int bid = get_group_id(0);

    // Load icnrCurrent from global icnr[bid] into local memory by first thread in workgroup
    if (thid == 0) {
        icnrCurrent = icnr[bid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);  // Ensure all threads see updated icnrCurrent

    int indexA = bid * blockSize + thid;
    int indexB = indexA + (blockSize / 2);

    g_idata[indexA] += icnrCurrent;
    g_idata[indexB] += icnrCurrent;

}

)CLC"};


template <class T>
OCLStream<T>::OCLStream(const int ARRAY_SIZE, const int device_index)
{
  if (!cached)
    getDeviceList();

  // Setup default OpenCL GPU
  if (device_index >= devices.size())
    throw std::runtime_error("Invalid device index");
  device = devices[device_index];

  // Determine sensible dot kernel NDRange configuration
  if (device.getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_CPU)
  {
    dot_num_groups = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    dot_wgsize     = device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE>() * 2;
  }
  else
  {
    dot_num_groups = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() * 4;
    dot_wgsize     = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  }

  // Print out device information
  std::cout << "Using OpenCL device " << getDeviceName(device_index) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;
  std::cout << "Reduction kernel config: " << dot_num_groups << " groups of size " << dot_wgsize << std::endl;

  context = cl::Context(device);
  queue = cl::CommandQueue(context);

  // Create program
  cl::Program program(context, kernels);
  std::ostringstream args;
  args << "-DstartScalar=" << startScalar << " ";
  if (sizeof(T) == sizeof(double))
  {
    args << "-DTYPE=double";
    // Check device can do double
    if (!device.getInfo<CL_DEVICE_DOUBLE_FP_CONFIG>())
      throw std::runtime_error("Device does not support double precision, please use --float");
    try
    {
      program.build(args.str().c_str());
    }
    catch (cl::Error& err)
    {
      if (err.err() == CL_BUILD_PROGRAM_FAILURE)
      {
        std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>()[0].second << std::endl;
        throw err;
      }
    }
  }
  else if (sizeof(T) == sizeof(float))
  {
    args << "-DTYPE=float";
    program.build(args.str().c_str());
  }

  // Create kernels
  init_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, T, T, T>(program, "init");
  copy_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer>(program, "copy");
  mul_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer>(program, "mul");
  add_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>(program, "add");
  triad_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>(program, "triad");
  nstream_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>(program, "nstream");
  dot_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl_int>(program, "stream_dot");
  scan_kernel = new cl::KernelFunctor<cl::Buffer, int, cl::Buffer, cl::LocalSpaceArg>(program, "scan");
  addBlockSums = new cl::KernelFunctor<cl::Buffer, cl::Buffer, int>(program, "addblockSums");



  array_size = ARRAY_SIZE;
  int scan_array_size = (int)(array_size * (1.0f - powf(1.0f / (1024*2), ceilf(logf(array_size) / logf((float)(1024*2))))) / (1.0f - 1.0f / (1024*2))) + 1;   


  // Check buffers fit on the device
  cl_ulong totalmem = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
  cl_ulong maxbuffer = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
  if (maxbuffer < sizeof(T)*ARRAY_SIZE)
    throw std::runtime_error("Device cannot allocate a buffer big enough");
  if (totalmem < 4*sizeof(T)*ARRAY_SIZE)
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");

  // Create buffers
  d_a = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * ARRAY_SIZE);
  d_b = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * ARRAY_SIZE);
  d_c = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * ARRAY_SIZE);
  d_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * scan_array_size);
  d_sum = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(T) * dot_num_groups);

  sums = std::vector<T>(dot_num_groups);
}

template <class T>
OCLStream<T>::~OCLStream()
{
  delete init_kernel;
  delete copy_kernel;
  delete mul_kernel;
  delete add_kernel;
  delete triad_kernel;
  delete nstream_kernel;
  delete dot_kernel;

  devices.clear();
}

template <class T>
void OCLStream<T>::copy()
{
  (*copy_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size)),
    d_a, d_c
  );
  queue.finish();
}

template <class T>
void OCLStream<T>::mul()
{
  (*mul_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size)),
    d_b, d_c
  );
  queue.finish();
}

template <class T>
void OCLStream<T>::add()
{
  (*add_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size)),
    d_a, d_b, d_c
  );
  queue.finish();
}

template <class T>
void OCLStream<T>::triad()
{
  (*triad_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size)),
    d_a, d_b, d_c
  );
  queue.finish();
}

template <class T>
void OCLStream<T>::nstream()
{
  (*nstream_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size)),
    d_a, d_b, d_c
  );
  queue.finish();
}

template <class T>
T OCLStream<T>::dot()
{
  (*dot_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(dot_num_groups*dot_wgsize), cl::NDRange(dot_wgsize)),
    d_a, d_b, d_sum, cl::Local(sizeof(T) * dot_wgsize), array_size
  );
  cl::copy(queue, d_sum, sums.begin(), sums.end());

  T sum{};
  for (T val : sums)
    sum += val;

  return sum;
}


template <class T>
void OCLStream<T>::recursiveScan(cl::Buffer &d_input, int current_length, int original_length) {
    const int B = 2048;  // Block size

    current_length = ((current_length & (current_length - 1)) != 0) ? 1 << (int)ceil(log2(current_length)) : current_length;
    int num_of_blocks = (current_length/B > 0) ? current_length/B : 1;

    int scan_array_size = (int)(current_length * (1.0f - powf(1.0f / (1024*2), ceilf(logf(current_length) / logf((float)(1024*2))))) / (1.0f - 1.0f / (1024*2)));  

    cl_buffer_region region;
    region.origin = (original_length == current_length) ? (original_length) * sizeof(T): (original_length+current_length) * sizeof(T);           
    region.size   = ((scan_array_size - current_length) > 0) ? (scan_array_size - current_length) * sizeof(T) : sizeof(T); // size of sub-buffer
    cl::Buffer sums = d_d.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region);


    int shared_size = (B + CONFLICT_FREE_OFFSET(B)) * sizeof(T);
    int sums_size = (num_of_blocks + CONFLICT_FREE_OFFSET(num_of_blocks)) * sizeof(T);

    (*scan_kernel)(
      cl::EnqueueArgs(queue, cl::NDRange(num_of_blocks * (B / 2)), cl::NDRange(B / 2)),
      d_input, current_length, sums, cl::Local(shared_size)
    );
    queue.finish();

    if (num_of_blocks > B){ //Recursive case
      recursiveScan(sums, num_of_blocks, original_length);

    } else if(num_of_blocks > 1) { //Base case
        cl::Buffer dummy_buffer(context, CL_MEM_READ_WRITE, num_of_blocks * sizeof(T));

        (*scan_kernel)(
          cl::EnqueueArgs(queue, cl::NDRange(num_of_blocks/2)),
          sums, num_of_blocks, sums, cl::Local(sums_size)
        );
        queue.finish();
    }

    if (num_of_blocks > 1)
    {
      (*addBlockSums)(
        cl::EnqueueArgs(queue, cl::NDRange(num_of_blocks * (B / 2)), cl::NDRange(B / 2)),
        d_input, sums, B
      );
      queue.finish();

    }
    
}

template <class T>
void OCLStream<T>::scan() {
  recursiveScan(d_d, array_size, array_size);
}


template <class T>
void OCLStream<T>::init_arrays(T initA, T initB, T initC)
{
  (*init_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size)),
    d_a, d_b, d_c, d_d, initA, initB, initC
  );
  queue.finish();
}

template <class T>
void OCLStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, std::vector<T>& d)
{
 try {
    int scan_array_size = (int)(array_size * (1.0f - powf(1.0f / (1024*2), ceilf(logf(array_size) / logf((float)(1024*2))))) / (1.0f - 1.0f / (1024*2))) + 1;   

    queue.enqueueReadBuffer(d_a, CL_TRUE, 0, sizeof(T)*array_size, a.data()); //major change
    queue.enqueueReadBuffer(d_b, CL_TRUE, 0, sizeof(T)*array_size, b.data());
    queue.enqueueReadBuffer(d_c, CL_TRUE, 0, sizeof(T)*array_size, c.data());
    queue.enqueueReadBuffer(d_d, CL_TRUE, 0, sizeof(T)*scan_array_size, d.data());
    queue.finish();
  } catch (cl::Error &e) {
      std::cerr << "OpenCL error during read_arrays: " << e.what() << " (" << e.err() << ")" << std::endl;
      throw;
  }


}

void getDeviceList(void)
{
  // Get list of platforms
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  // Enumerate devices
  for (unsigned i = 0; i < platforms.size(); i++)
  {
    std::vector<cl::Device> plat_devices;
    platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &plat_devices);
    devices.insert(devices.end(), plat_devices.begin(), plat_devices.end());
  }
  cached = true;
}

void listDevices(void)
{
  getDeviceList();

  // Print device names
  if (devices.size() == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < devices.size(); i++)
    {
      std::cout << i << ": " << getDeviceName(i) << std::endl;
    }
    std::cout << std::endl;
  }


}

std::string getDeviceName(const int device)
{
  if (!cached)
    getDeviceList();

  std::string name;
  cl_device_info info = CL_DEVICE_NAME;

  if (device < devices.size())
  {
    devices[device].getInfo(info, &name);
  }
  else
  {
    throw std::runtime_error("Error asking for name for non-existant device");
  }

  return name;

}

std::string getDeviceDriver(const int device)
{
  if (!cached)
    getDeviceList();

  std::string driver;

  if (device < devices.size())
  {
    devices[device].getInfo(CL_DRIVER_VERSION, &driver);
  }
  else
  {
    throw std::runtime_error("Error asking for driver for non-existant device");
  }

  return driver;
}


template class OCLStream<float>;
template class OCLStream<double>;
