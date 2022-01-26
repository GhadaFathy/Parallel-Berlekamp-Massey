#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h> 

#define NUMTHREAD 64
#define GRIDWIDTH 128
#define bufferSize 4096

char buffer[bufferSize];
int deviceNum = 1, debug = 0, randGen = 0;

int lenS, lenBitS, lenLastZero, power2[32];
int *S, *bitS, *rBitS, *C, *T, *lastZero, *B, *print, *print1;
int x=1, L=0, d = 0;

unsigned int bitMask[32] = {
  0x00000000, 0x80000000, 0xC0000000, 0xE0000000,
  0xF0000000, 0xF8000000, 0xFC000000, 0xFE000000,
  0xFF000000, 0xFF800000, 0xFFC00000, 0xFFE00000,
  0xFFF00000, 0xFFF80000, 0xFFFC0000, 0xFFFE0000,
  0xFFFF0000, 0xFFFF8000, 0xFFFFC000, 0xFFFFE000,
  0xFFFFF000, 0xFFFFF800, 0xFFFFFC00, 0xFFFFFE00,
  0xFFFFFF00, 0xFFFFFF80, 0xFFFFFFC0, 0xFFFFFFE0,
  0xFFFFFFF0, 0xFFFFFFF8, 0xFFFFFFFC, 0xFFFFFFFE,
};

int *gpuLenLastZero, gpuN, *gpuD;
int *gpuS, *gpuRBitS, *gpuC, *gpuB, *gpuT, *gpuLastZero, *gpuPower2,*gpuprint,*gpuprint1;
unsigned int *gpuBitMask;

void printC(int length){
  int i, j, bitPos, tmp;

  printf("C= ");
  bitPos = 31;
  j = 0;
  for(i=0;i<length;i++){
    tmp = C[j] & power2[bitPos];
    printf("%1d",tmp?1:0);
    bitPos--;
    if (bitPos == -1){
      bitPos = 31;
      j++;
    }
  }
  printf("\n");
}
void printB(int length){
  int i, j, bitPos, tmp;

  printf("B= ");
  bitPos = 31;
  j = 0;
  for(i=0;i<length;i++){
    tmp = B[j] & power2[bitPos];
    printf("%1d",tmp?1:0);
    bitPos--;
    if (bitPos == -1){
      bitPos = 31;
      j++;
    }
  }
  printf("\n");
}
void printPrint(int length){
  int i, j, bitPos, tmp;

  printf("Print= ");
  bitPos = 31;
  j = 0;
  for(i=0;i<length;i++){
    tmp = print[j] & power2[bitPos];
    printf("%1d",tmp?1:0);
    bitPos--;
    if (bitPos == -1){
      bitPos = 31;
      j++;
    }
  }
  printf("\n");
}
void printPrint1(int length){
  int i, j, bitPos, tmp;

  printf("Print1= ");
  bitPos = 31;
  j = 0;
  for(i=0;i<length;i++){
    tmp = print1[j] & power2[bitPos];
    printf("%1d",tmp?1:0);
    bitPos--;
    if (bitPos == -1){
      bitPos = 31;
      j++;
    }
  }
  printf("\n");
}
void initGPU(void){
  cudaSetDevice(deviceNum);
  cudaMalloc((void**)&gpuS,sizeof(int)*lenS);
  cudaMalloc((void**)&gpuRBitS,sizeof(int)*(lenBitS+1));
  cudaMalloc((void**)&gpuC,sizeof(int)*lenBitS);
  cudaMalloc((void**)&gpuB,sizeof(int)*lenBitS);
  cudaMalloc((void**)&gpuT,sizeof(int)*lenS);
  cudaMalloc((void**)&gpuprint,sizeof(int)*lenS);
  cudaMalloc((void**)&gpuprint1,sizeof(int)*lenS);
  cudaMalloc((void**)&gpuLastZero,sizeof(int)*(lenBitS+1));
  cudaMalloc((void**)&gpuLenLastZero,sizeof(int)*1);
  cudaMalloc((void**)&gpuD,sizeof(int)*1);
  cudaMalloc((void**)&gpuPower2,sizeof(int)*32);
  cudaMalloc((void**)&gpuBitMask,sizeof(unsigned int)*32);
}

void init(int argc, char* argv[]){
  int i, j, bitPos;
  FILE *fp;
  char* token;

  if (argc < 3){
    printf("Usage: ./bit filename length -d deviceNum -b debugLvl\n");
    printf("\tto generate random string: ./bit randGen length\n");
    exit(1);
  }
  if (argc > 3){
    i = 3;
    while (i<argc){
      if (!strcmp(argv[i],"-d")) sscanf(argv[i+1],"%d",&deviceNum);
      if (!strcmp(argv[i],"-b")) sscanf(argv[i+1],"%d",&debug);
      i += 2;
    }
  }
  if (strcmp(argv[1],"randGen") == 0) randGen = 1;
  else{
    randGen = 0;
    fp = fopen(argv[1],"r");
    if (!fp){
      printf("%s doesn't exist\n",argv[1]);
      exit(1);
    }
  }
  sscanf(argv[2],"%d",&lenS);
  if (lenS < 1){
    printf("positive length needed\n");
    exit(1);
  }
  power2[0] = 1;
  for(i=1;i<32;i++) power2[i] = 2*power2[i-1];

  lenBitS = (lenS+31)/32;
  S = (int*)malloc(sizeof(int)*lenS);
  bitS = (int*)malloc(sizeof(int)*lenBitS);
  rBitS = (int*)malloc(sizeof(int)*(lenBitS+1));
  C = (int*)malloc(sizeof(int)*lenBitS);
  B = (int*)malloc(sizeof(int)*lenBitS);
  T = (int*)malloc(sizeof(int)*lenS);
  print = (int*)malloc(sizeof(int)*lenS);
  print1 = (int*)malloc(sizeof(int)*lenS);
  lastZero = (int*)malloc(sizeof(int)*(lenBitS+1));
  lenLastZero = 1;

  if (randGen){
    S[0] = 1;
    j = 0;
    bitS[j] = 1;
    bitPos = 1;
    for(i=1;i<lenS;i++){
      S[i] = rand()%2;
      if (S[i])	bitS[j] |= power2[bitPos];
      bitPos++;
      if (bitPos == 32){
	bitPos = 0;
	j++;
	bitS[j] = 0;
      }
    }
  }
  else{
    i = 0;
    j = 0;
    bitS[j] = 0;
    bitPos = 0;
    while (fgets(buffer,bufferSize,fp) && i<lenS){
      token = strtok(buffer," ");
      while ((token) && i<lenS){
	S[i] = atoi(token);
	if (S[i]) bitS[j] |= power2[bitPos];
	bitPos++;
	if (bitPos == 32){
	  bitPos = 0;
	  j++;
	  bitS[j] = 0;
	}
	i++;
	token = strtok(NULL," ");
      }
    }
    fclose(fp);
    if (i != lenS){
      printf("file has only %d bits\n",i);
      exit(1);
    }
  }
  for(i=0;i<lenBitS;i++){
    rBitS[lenBitS-i-1] = bitS[i];
    C[i] = lastZero[i] = print[i] = print1[i] = 0;
  }
  rBitS[lenBitS] = lastZero[lenBitS] = 0;
  C[0] = power2[31];
  for(i=0;i<lenS;i++) T[i] = 0;
}

__global__ void calcD_kernel(int* d, int* T, int numRow, int N) {
	
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int tid = threadIdx.x + bid* blockDim.x;
	if(tid == 0)
		d[0] = T[N-numRow];
	
	__syncthreads();
}

__global__ void
kernel1(int* rBitS, int* C, int* T, int lenS, int numRow, int numCol,
	unsigned int* bitMask,int col, int * print){
  __shared__ unsigned int sharedT[NUMTHREAD];
  int numRowBit = (numRow+31)/32;
  int myCol = blockIdx.y*gridDim.x + blockIdx.x;
  int myC = threadIdx.x;
  int myS = (lenS - col - numRow)/32 + threadIdx.x;
  int shiftS = (lenS - col - numRow) % 32;
  unsigned int currS, nextS;

  if (myCol >= 1) return;
  //the whole block of threads return because the column is not needed

  sharedT[threadIdx.x] = 0;
  //if(myCol==0) print[threadIdx.x]=myS;
  while (myC < numRowBit){
    currS = rBitS[myS];
    //the 32-bit S may stride over two int, so frame shift
    if (shiftS){
      currS <<= shiftS;
      nextS = rBitS[myS+1];
      nextS >>= 32 - shiftS;
      currS |= nextS;
    }
    sharedT[threadIdx.x] ^= (C[myC] & currS);
    myC += NUMTHREAD;
    myS += NUMTHREAD;
  }
  
  __syncthreads();

  //reduction
  if (NUMTHREAD >= 1024){
    if (threadIdx.x < 512) sharedT[threadIdx.x] ^= sharedT[threadIdx.x+512];
    __syncthreads();
  }
  if (NUMTHREAD >= 512){
    if (threadIdx.x < 256) sharedT[threadIdx.x] ^= sharedT[threadIdx.x+256];
    __syncthreads();
  }
  if (NUMTHREAD >= 256){
    if (threadIdx.x < 128) sharedT[threadIdx.x] ^= sharedT[threadIdx.x+128];
    __syncthreads();
  }
  if (NUMTHREAD >= 128){
    if (threadIdx.x < 64) sharedT[threadIdx.x] ^= sharedT[threadIdx.x+64];
    __syncthreads();
  }
  if (threadIdx.x < 32){
    volatile unsigned int *tmem = sharedT;
    if (NUMTHREAD >= 64) tmem[threadIdx.x] ^= tmem[threadIdx.x+32];
    if (NUMTHREAD >= 32) tmem[threadIdx.x] ^= tmem[threadIdx.x+16];
    if (NUMTHREAD >= 16) tmem[threadIdx.x] ^= tmem[threadIdx.x+8];
    if (NUMTHREAD >= 8) tmem[threadIdx.x] ^= tmem[threadIdx.x+4];
    if (NUMTHREAD >= 4) tmem[threadIdx.x] ^= tmem[threadIdx.x+2];
    if (NUMTHREAD >= 2) tmem[threadIdx.x] ^= tmem[threadIdx.x+1];
  }

  if (threadIdx.x == 0){
    //count the # of 1 bits in sharedT[0]
    currS = sharedT[0];
    currS = currS - ((currS >> 1) & 0x55555555);
    currS = (currS & 0x33333333) + ((currS >> 2) & 0x33333333);
    currS = ((currS + (currS >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
    T[0] = currS%2;
  }
}

__global__ void
kernel2(int* C, int* T, int* lastZero, int* lenLastZero, int numRow,
	int* power2){
  int myId = blockIdx.x*blockDim.x + threadIdx.x;
  int whichC, whichBit;

  if (myId >= (numRow+31)/32) return;
  if (T[0]){
    if (myId == 0){
      whichC = (numRow-1) / 32;
      whichBit = 31 - (numRow-1) % 32;
      C[whichC] |= power2[whichBit];
    }
    return;
  }
  lastZero[myId] = C[myId];
  if (myId == 0) *lenLastZero = numRow;
}

//just reduction
__global__ void kernel3(int* T, int numCol){
  __shared__ int sharedT[NUMTHREAD];
  int myId = blockIdx.x*blockDim.x + threadIdx.x;

  sharedT[threadIdx.x] = (myId < numCol) ? T[myId] : 0;
  __syncthreads();
  if (NUMTHREAD >= 1024){
    if (threadIdx.x < 512)
      sharedT[threadIdx.x] = sharedT[threadIdx.x]+sharedT[threadIdx.x+512];
    __syncthreads();
  }
  if (NUMTHREAD >= 512){
    if (threadIdx.x < 256)
      sharedT[threadIdx.x] = sharedT[threadIdx.x]+sharedT[threadIdx.x+256];
    __syncthreads();
  }
  if (NUMTHREAD >= 256){
    if (threadIdx.x < 128)
      sharedT[threadIdx.x] = sharedT[threadIdx.x]+sharedT[threadIdx.x+128];
    __syncthreads();
  }
  if (NUMTHREAD >= 128){
    if (threadIdx.x < 64)
      sharedT[threadIdx.x] = sharedT[threadIdx.x]+sharedT[threadIdx.x+64];
    __syncthreads();
  }
  if (threadIdx.x < 32){
    volatile int *tmem = sharedT;
    if (NUMTHREAD >= 64)
      tmem[threadIdx.x] = tmem[threadIdx.x]+tmem[threadIdx.x+32];
    if (NUMTHREAD >= 32)
      tmem[threadIdx.x] = tmem[threadIdx.x]+tmem[threadIdx.x+16];
    if (NUMTHREAD >= 16)
      tmem[threadIdx.x] = tmem[threadIdx.x]+tmem[threadIdx.x+8];
    if (NUMTHREAD >= 8)
      tmem[threadIdx.x] = tmem[threadIdx.x]+tmem[threadIdx.x+4];
    if (NUMTHREAD >= 4)
      tmem[threadIdx.x] = tmem[threadIdx.x]+tmem[threadIdx.x+2];
    if (NUMTHREAD >= 2)
      tmem[threadIdx.x] = tmem[threadIdx.x]+tmem[threadIdx.x+1];
  }
  if (threadIdx.x == 0) T[myId] = sharedT[0];
}
// ===================================================================================
__global__ void kernel4(int* C, int* B, int shiftLZ, int numThread, int * print, int * print1){
  int myId = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int myLZ, tmp;

  if (myId >= numThread) return;
  myLZ = B[myId];
  if (shiftLZ) {
	  if(myId == 0) {
		// shift right >>
		myLZ >>= shiftLZ;
	  } else {
		tmp = B[myId-1];
		myLZ >>= shiftLZ;
		tmp <<= 32 - shiftLZ;
		myLZ |= tmp;
	  }
  }
  print[myId] = myLZ;
  print1[myId] = C[myId];
  C[myId] ^= myLZ;
}
// ===================================================================================
__global__ void kernel4_updateB(int* C, int* B, int *lenLastZero, int shiftLZ, int numThread, int offsetC, int numRow, int * print, int * print1){
	
	__shared__ unsigned int sharedT[NUMTHREAD];
	
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int myId = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int myLZ, tmp;

	if (myId >= numThread) return;

	sharedT[threadIdx.x] = C[myId];

	myLZ = B[myId];
	if (shiftLZ) {
	  if(myId == 0) {
		// shift right >>
		myLZ >>= shiftLZ;
	  } else {
		tmp = B[myId-1];
		myLZ >>= shiftLZ;
		tmp <<= 32 - shiftLZ;
		myLZ |= tmp;
	  }
  }
  
  print[myId] = myLZ;
  print1[myId] = C[myId+offsetC];
	C[myId+offsetC] ^= myLZ;

	__syncthreads();
	B[threadIdx.x - bid] = sharedT[threadIdx.x];
	if (myId == 0) lenLastZero[0] = numRow;
}
// ===================================================================================

__global__ void
kernel5(int* C, int* S, int* T,	int numRow, int numCol, int* power2){
  int myId = blockIdx.x*blockDim.x + threadIdx.x;
  int whichC, whichBit;

  if (T[0] == 0) return;
  if (myId >= numCol) return;
  if (myId == 0){
    whichC = (numRow-1) / 32;
    whichBit = 31 - (numRow-1)%32;
    C[whichC] |= power2[whichBit];
  }
  T[myId] = (T[myId]+S[myId])%2;
}

void callKernels(void){
  int numRow, numCol, numBlock, prevNumBlock, halfway;
  int numThread, offsetC, shiftB;

  cudaMemcpy(gpuS,S,sizeof(int)*lenS,cudaMemcpyHostToDevice);
  cudaMemcpy(gpuRBitS,rBitS,sizeof(int)*(lenBitS+1),cudaMemcpyHostToDevice);
  cudaMemcpy(gpuC,C,sizeof(int)*lenBitS,cudaMemcpyHostToDevice);
  cudaMemcpy(gpuB,C,sizeof(int)*lenBitS,cudaMemcpyHostToDevice);
  cudaMemcpy(gpuT,T,sizeof(int)*lenS,cudaMemcpyHostToDevice);
  cudaMemcpy(gpuprint,print,sizeof(int)*lenS,cudaMemcpyHostToDevice);
  cudaMemcpy(gpuprint1,print1,sizeof(int)*lenS,cudaMemcpyHostToDevice);
  cudaMemcpy(gpuPower2,power2,sizeof(int)*32,cudaMemcpyHostToDevice);
  cudaMemcpy(gpuBitMask,bitMask,sizeof(unsigned int)*32,
	     cudaMemcpyHostToDevice);
  cudaMemcpy(gpuLastZero,lastZero,sizeof(int)*(lenBitS+1),
	     cudaMemcpyHostToDevice);
  gpuLastZero++;
  cudaMemcpy(gpuLenLastZero,&lenLastZero,sizeof(int)*1,cudaMemcpyHostToDevice);

  halfway = (lenS+1)/2;
  //==========================================================================
  int N=0;
  numRow = 2;

  d = S[0];
  if(d == 0) x++;
  else L=1;
  
  // NOTE: numRow is currentt matrix size
  for(N=1; N<lenS; N++) {
     /* for(int i=0;i<lenBitS;i++){
          //	print[i] = print1[i] = 0;
      }*/
      
	  // calcuate numBlock
	  if (numRow <= halfway) {
		  numCol = numRow;
		  numBlock = (numCol+GRIDWIDTH-1)/GRIDWIDTH;
	  } else {
		  numCol = (lenS - numRow) + 1;
		  numBlock = (numCol+GRIDWIDTH-1)/GRIDWIDTH;	
	  }
       
	  // Multiplay
	  int col = N - (numRow-1);
	  kernel1<<<dim3(numBlock,GRIDWIDTH),NUMTHREAD>>>(gpuRBitS, gpuC, gpuT, lenS, numRow, numCol, gpuBitMask,col,gpuprint);
	  cudaDeviceSynchronize();
      cudaMemcpy(&d,(gpuT),sizeof(int)*1,cudaMemcpyDeviceToHost);

	  // d diversions
	  if (d == 0) {
		  x++;
		  numRow++;
	  }  else if (d != 0 && (2*L) > N) {
	      
		  numThread = (numRow/32) + ((numRow%32) ? 1 : 0);
		
		  shiftB = x;
		  numBlock = (numThread+NUMTHREAD-1)/NUMTHREAD;
		  //offsetC = (numRow+31)/32 - numThread;
		  offsetC = x/32 ;
		 
		  kernel4<<<numBlock,NUMTHREAD>>>(gpuC+offsetC,gpuB,shiftB,numThread,gpuprint,gpuprint1);
		  cudaDeviceSynchronize();
		  x++;
		  numRow++;
	  }else if (d != 0 && (2*L) <= N) {
		  // call kernel
		  //int xx = numRow - lenLastZero;
		  numThread = (numRow/32) + ((numRow%32) ? 1 : 0);
		  shiftB = x;
		 
		  numBlock = (numThread+NUMTHREAD-1)/NUMTHREAD;
		  offsetC = x/32 ;
		 
		  kernel4_updateB<<<numBlock,NUMTHREAD>>>(gpuC,gpuB,gpuLenLastZero,shiftB,numThread, offsetC,numRow,gpuprint,gpuprint1);
		  cudaDeviceSynchronize();
		  cudaMemcpy(&lenLastZero,gpuLenLastZero,sizeof(int)*1,cudaMemcpyDeviceToHost);
		  L = N+1-L;
		  x=1;
	  }
  	
  }

  //==========================================================================
  
  cudaMemcpy(C,gpuC,sizeof(int)*lenBitS,cudaMemcpyDeviceToHost);
  
}

int main(int argc, char *argv[]){
  StopWatchInterface *timer = 0;
 
  init(argc,argv);
  initGPU();

  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  sdkResetTimer(&timer);
  callKernels();
  sdkStopTimer(&timer);

  if (debug)
  {
      printC(lenS);
	 
  }
  printf("device %d, numT %d, length %d, time %.3f s\n",
	 deviceNum,NUMTHREAD,lenS,sdkGetTimerValue(&timer)/1000);
  return 0;
}
