// Copyright (c) 2021 Tom Deakin and Tom Lin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "STDIndicesStream.h"

#ifndef ALIGNMENT
#define ALIGNMENT (2*1024*1024) // 2MB
#define TBSIZE (1024);
#endif

template <class T>
STDIndicesStream<T>::STDIndicesStream(const int ARRAY_SIZE, int device)
noexcept :
  array_size{ARRAY_SIZE},
  scan_array_size{
    static_cast<int>(
      ARRAY_SIZE * (1.0f - powf(1.0f / (1024 * 2), ceilf(logf(ARRAY_SIZE) / logf(static_cast<float>(1024 * 2)))))
      / (1.0f - 1.0f / (1024 * 2))
    )
  },
  range(0, array_size),
  a(alloc_raw<T>(ARRAY_SIZE)),
  b(alloc_raw<T>(ARRAY_SIZE)),
  c(alloc_raw<T>(ARRAY_SIZE)),
  d(alloc_raw<T>(scan_array_size))
{
    std::cout << "Backing storage typeid: " << typeid(a).name() << std::endl;
#ifdef USE_ONEDPL
    std::cout << "Using oneDPL backend: ";
#if ONEDPL_USE_DPCPP_BACKEND
    std::cout << "SYCL USM (device=" << exe_policy.queue().get_device().get_info<sycl::info::device::name>() << ")";
#elif ONEDPL_USE_TBB_BACKEND
    std::cout << "TBB " TBB_VERSION_STRING;
#elif ONEDPL_USE_OPENMP_BACKEND
    std::cout << "OpenMP";
#else
    std::cout << "Default";
#endif
    std::cout << std::endl;
#endif
}

template<class T>
STDIndicesStream<T>::~STDIndicesStream() {
  dealloc_raw(a);
  dealloc_raw(b);
  dealloc_raw(c);
  dealloc_raw(d);
}

template <class T>
void STDIndicesStream<T>::init_arrays(T initA, T initB, T initC)
{
  std::fill(exe_policy, a, a + array_size, initA);
  std::fill(exe_policy, b, b + array_size, initB);
  std::fill(exe_policy, c, c + array_size, initC);
  std::iota(d, d + array_size, T(0));
}


template <class T>
void STDIndicesStream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c, std::vector<T>& h_d)
{

  int scan_array_size = (int)(array_size * (1.0f - powf(1.0f / (1024*2), ceilf(logf(array_size) / logf((float)(1024*2))))) / (1.0f - 1.0f / (1024*2)));   

  
  std::copy(a, a + array_size, h_a.begin());
  std::copy(b, b + array_size, h_b.begin());
  std::copy(c, c + array_size, h_c.begin());
  std::copy(d, d + scan_array_size, h_d.begin());
}

template <class T>
void STDIndicesStream<T>::copy()
{
  // c[i] = a[i]
  std::copy(exe_policy, a, a + array_size, c);
}

template <class T>
void STDIndicesStream<T>::mul()
{
  //  b[i] = scalar * c[i];
  std::transform(exe_policy, range.begin(), range.end(), b, [c = this->c, scalar = startScalar](int i) {
    return scalar * c[i];
  });
}

template <class T>
void STDIndicesStream<T>::add()
{
  //  c[i] = a[i] + b[i];
  std::transform(exe_policy, range.begin(), range.end(), c, [a = this->a, b = this->b](int i) {
    return a[i] + b[i];
  });
}

template <class T>
void STDIndicesStream<T>::triad()
{
  //  a[i] = b[i] + scalar * c[i];
  std::transform(exe_policy, range.begin(), range.end(), a, [b = this->b, c = this->c, scalar = startScalar](int i) {
    return b[i] + scalar * c[i];
  });
}

template <class T>
void STDIndicesStream<T>::nstream()
{
  //  a[i] += b[i] + scalar * c[i];
  //  Need to do in two stages with C++11 STL.
  //  1: a[i] += b[i]
  //  2: a[i] += scalar * c[i];
  std::transform(exe_policy, range.begin(), range.end(), a, [a = this->a, b = this->b, c = this->c, scalar = startScalar](int i) {
    return a[i] + b[i] + scalar * c[i];
  });
}
   

template <class T>
T STDIndicesStream<T>::dot()
{
  // sum = 0; sum += a[i]*b[i]; return sum;
  return std::transform_reduce(exe_policy, a, a + array_size, b, T{});
}

template <class T>
void STDIndicesStream<T>::scan()
{
  std::inclusive_scan(
    exe_policy, d, d + array_size, d                         
  );  

}


void listDevices(void)
{
  std::cout << "Listing devices is not supported by the Parallel STL" << std::endl;
}

std::string getDeviceName(const int)
{
  return std::string("Device name unavailable");
}

std::string getDeviceDriver(const int)
{
  return std::string("Device driver unavailable");
}
template class STDIndicesStream<float>;
template class STDIndicesStream<double>;
