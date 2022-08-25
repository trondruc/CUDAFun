
#include <curand_kernel.h>

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <iostream>
using namespace std;

#define PI 3.1415926535897931f

// Variables needed in pi().
int numThreads; int numBlocks;
int threadNumberOfSamples;

// Hits within the unit circle.
int* m_hits;
// Random number generator on device. 
curandState* d_state;

int howMany = 30e6;

// Error checking.
inline void CHECK_ERR( cudaError_t err ) {
    if( err != cudaSuccess ) {
        const char* errStr = cudaGetErrorString( err );
        printf( "%s (%d): %s\n", __FILE__, __LINE__, errStr );
        assert( 0 );
    }
}

// Sequences generated with the same seed and different sequence 
// numbers will not have statistically correlated values.
__global__ void initRandomNumberGenerator( curandState* state ){
    int index = threadIdx.x + blockIdx.x * blockDim.x;       
    curand_init( 1234, index, 12, &state[index] );
}

void pi_init() {
  int device; int nProc;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute( &nProc, cudaDevAttrMultiProcessorCount, device );
 
  // Use T4 architecture optimized values.
  numThreads = 32;
  numBlocks = nProc * 10; 

  CHECK_ERR( cudaMallocManaged( &m_hits, sizeof(int) ) );
  cudaMemset( m_hits, 0, sizeof(int) );

  // Use one random number generator pr. thread.
  CHECK_ERR( cudaMalloc( &d_state, sizeof(curandState) * numThreads * numBlocks ) );

  initRandomNumberGenerator<<< numBlocks, numThreads >>>( d_state );
  CHECK_ERR( cudaGetLastError() );
  
  threadNumberOfSamples = ceil( double(howMany) / ( numThreads * numBlocks ) );
}

// Generates two floats on [0,1] and tests (x,y) within the unit circle.
__device__ inline void generateRandomAndUpdate( int& counter, curandState& state ) {
  float x = curand_uniform( &state );
  float y = curand_uniform( &state );

  if( x * x + y * y < 1.f ) 
    ++counter;
}

// The calculation of Pi using Monte Carlo doesn't have correlation between threads.
__global__ void monteCarloPi( int* count, curandState* state, const int samples, const int total ) {
  extern __shared__ int smem[];

  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  int globalIndex = index * samples;

  // The number of unit circle hits for this thread.
  int localCounter = 0;
  // Random number generator.
  auto localState = state[index];

  // Each thread takes a portion of samples.
  for( auto i = 0; i < samples && globalIndex + i < total; ++i )
    generateRandomAndUpdate( localCounter, localState );

  // Prepare to sum threads in this block.
  smem[threadIdx.x] = localCounter; // 0 bank conflicts

  __syncthreads();

  // 1 thread handles the block contribution.
  if( threadIdx.x == 0 ){
    localCounter = thrust::reduce( thrust::seq, smem, smem + blockDim.x );
    atomicAdd( count, localCounter );
  }   
	
  state[index] = localState; 
}

// Tesla T4 speedup: a factor of ~3000 over 1 CPU for comparable accuracy.
double pi() {
    *m_hits = 0;
    monteCarloPi<<< numBlocks, numThreads,
                    sizeof(int) * numThreads >>>( m_hits, d_state, threadNumberOfSamples, howMany );
  
    CHECK_ERR( cudaGetLastError() );
    CHECK_ERR( cudaStreamSynchronize(0) );

    double myPi = 4.0 * (*m_hits) / howMany;
    return myPi;
}

void pi_reset() {
    CHECK_ERR( cudaFree( d_state ) );
    CHECK_ERR( cudaFree( m_hits ) );
    d_state = nullptr;
    m_hits = nullptr;
}

int main( int argc, char* argv[] ) {
    pi_init();
    cout << fabs( PI - pi() ) << endl;
    pi_reset();

    return 0;
}