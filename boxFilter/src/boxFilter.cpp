#include "boxFilter.h"
#include <chrono>
#include <random>
#include <iostream>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON
//计算量：h*w*(2*radius+1)*(2*radius+1)
void BoxFilter::filter(float *input, int radius, int height, int width, float *output) {
  for (int h = 0; h < height; ++h) {
    int height_sift = h * width;
    for (int w = 0; w < width; ++w) {
      int start_h = std::max(0, h - radius);
      int end_h = std::min(height - 1, h + radius);
      int start_w = std::max(0, w - radius);
      int end_w = std::min(width - 1, w + radius);

      float tmp = 0;
      for (int sh = start_h; sh <= end_h; ++sh) {
        for (int sw = start_w; sw <= end_w; ++ sw) {
          tmp += input[sh * width + sw];
        }
      }

      output[height_sift + w] = tmp;
    }
  }
}
//计算量：h*w*(2*radius+1)*2: 空间换时间
void BoxFilter::fastFilter(float *input, int radius, int height, int width, float *output) {
  float *cachePtr = &(cache[0]);
  // sum horizonal
  for (int h = 0; h < height; ++h) {
    int sift = h * width;
    for (int w = 0; w < width; ++w) {
      int start_w = std::max(0, w - radius);
      int end_w = std::min(width - 1, w + radius);

      float tmp = 0;
      for (int sw = start_w; sw <= end_w; ++ sw) {
        tmp += input[sift + sw];
      }

      cachePtr[sift + w] = tmp;
    }
  }

  // sum vertical
  for (int h = 0; h < height; ++h) {
    int shift = h * width;
    int start_h = std::max(0, h - radius);
    int end_h = std::min(height - 1, h + radius);

    for (int sh = start_h; sh <= end_h; ++sh) {
      int out_shift = sh * width;
      for (int w = 0; w < width; ++w) {//注意这边w是0到width
        output[out_shift + w] += cachePtr[shift + w];
      }
    }
  }
}
//计算量：h*w: 空间换时间
void BoxFilter::fastFilterV2(float *input, int radius, int height, int width, float *output) {
  float *cachePtr = &(cache[0]);
  // sum horizonal
  for (int h = 0; h < height; ++h) {
    int shift = h * width;
    //suppose :height=width=5, radius=1,so padding=1, kernel_size=2*radius + 1
    float tmp = 0;
    for (int w = 0; w < radius; ++w) {
      tmp += input[shift + w];//tmp=input[h][0];
    }

    for (int w = 0; w <= radius; ++w) {
      tmp += input[shift + w + radius];//tmp+=input[h][1];tmp+=input[h][1]+input[h][2];
      cachePtr[shift + w] = tmp;
    }
    //cachePtr[h][0]=input[h][0]+input[h][1];cachePtr[h][1]=input[h][0]+input[h][1]+input[h][2];

    int start = radius + 1;//2~3
    int end = width - 1 - radius;
    for (int w = start; w <= end; ++w) {
      tmp += input[shift + w + radius];
      tmp -= input[shift + w - radius - 1];
      cachePtr[shift + w] = tmp;//cachePtr[h][2],cachePtr[h][3]
    }
    //w方向的最后一列
    start = width - radius;//4
    for (int w = start; w < width; ++w) {
      tmp -= input[shift + w - radius - 1];//input[shift + w - radius - 1]：表示上一个kernel的第一个
      cachePtr[shift + w] = tmp;//这边又特殊处理了最后一个4: cachePtr[h][4]
    }
  }
  //上述代码中定义了一个ColSum用于保存每行某个位置处在列方向上指定半径内各中间元素的累加值，对于第一行，做特殊处理，其他行的每个元素的处理方式和
  float *colSumPtr = &(colSum[0]);
  for (int indexW = 0; indexW < width; ++indexW) {
    colSumPtr[indexW] = 0;
  } 
  // sum vertical,radius=1;
  for (int h = 0; h < radius; ++h) {
    int shift = h * width;
    for (int w = 0; w < width; ++w) {
      colSumPtr[w] += cachePtr[shift + w];//cachePtr[h][w]
    }
  }
  //这边h仅有一个数字=0
  //colSumPtr[0] = cachePtr[0][0]
  //colSumPtr[1] = cachePtr[0][1]
  //这边h=0或者1。。。。
  for (int h = 0; h <= radius; ++h) {
    float *addPtr = cachePtr + (h + radius) * width;//cachePtr[h+radius]
    int shift = h * width;
    float *outPtr = output + shift; //output[h]
    for (int w = 0; w < width; ++w) {
      colSumPtr[w] += addPtr[w];//colSumPtr[w] += cachePtr[h+radius][w];  colSumPtr[w] += cachePtr[1][w]+cachePtr[2][w]
      outPtr[w] = colSumPtr[w];
    }
  }
  //当h=0的时候，output[h][w]=colSumPtr[w]；output[0][w]=colSumPtr[w]=cachePtr[1][w]+cachePtr[0][0]
  //当h=1的时候，output[h][w]=colSumPtr[w]；output[1][w]=colSumPtr[w]=cachePtr[1][w]+cachePtr[0][0]+cachePtr[2][w]

  int start = radius + 1;
  int end = height - 1 - radius;
  for (int h = start; h <= end; ++h) {
    float *addPtr = cachePtr + (h + radius) * width;
    float *subPtr = cachePtr + (h - radius - 1) * width;
    int shift = h * width;
    float *outPtr = output + shift;
    for (int w = 0; w < width; ++w) {
      colSumPtr[w] += addPtr[w];
      colSumPtr[w] -= subPtr[w];
      outPtr[w] = colSumPtr[w];
    }
  }
  //这边是处理最后一行的逻辑
  start = height - radius;
  for (int h = start; h < height; ++h) {
    float *subPtr = cachePtr + (h - radius - 1) * width;
    int shift = h * width;
    float *outPtr = output + shift; 
    for (int w = 0; w < width; ++w) {
      colSumPtr[w] -= subPtr[w];
      outPtr[w] = colSumPtr[w];
    }
  }
}

void BoxFilter::fastFilterV2NeonIntrinsics(float *input, int radius, int height, int width, float *output) {
  float *cachePtr = &(cache[0]);
  // sum horizonal
  for (int h = 0; h < height; ++h) {
    int shift = h * width;

    float tmp = 0;
    for (int w = 0; w < radius; ++w) {
      tmp += input[shift + w];
    }

    for (int w = 0; w <= radius; ++w) {
      tmp += input[shift + w + radius];
      cachePtr[shift + w] = tmp;
    }

    int start = radius + 1;
    int end = width - 1 - radius;
    for (int w = start; w <= end; ++w) {
      tmp += input[shift + w + radius];
      tmp -= input[shift + w - radius - 1];
      cachePtr[shift + w] = tmp;
    }

    start = width - radius;
    for (int w = start; w < width; ++w) {
      tmp -= input[shift + w - radius - 1];
      cachePtr[shift + w] = tmp;
    }
  }

  float *colSumPtr = &(colSum[0]);

  for (int indexW = 0; indexW < width; ++indexW) {
    colSumPtr[indexW] = 0;
  } 

  int n = width >> 2;
  int re = width - (n << 2);
  // sum vertical
  for (int h = 0; h < radius; ++h) {
    int shift = h * width;

    float *tmpColSumPtr = colSumPtr;
    float *tmpCachePtr = cachePtr + shift;
    int indexW = 0;

    int nn = n;
    int remain = re;
#if __ARM_NEON
    for (; nn > 0; nn--) {
      float32x4_t _colSum = vld1q_f32(tmpColSumPtr);
      float32x4_t _cache = vld1q_f32(tmpCachePtr);

      float32x4_t _tmp = vaddq_f32(_colSum, _cache);

      vst1q_f32(tmpColSumPtr, _tmp); 

      tmpColSumPtr += 4;
      tmpCachePtr += 4;
    }
#endif // __ARM_NEON
    for (; remain > 0; --remain) {
      *tmpColSumPtr += *tmpCachePtr;
      tmpColSumPtr ++;
      tmpCachePtr ++;
    }
  }

  for (int h = 0; h <= radius; ++h) {
    float *addPtr = cachePtr + (h + radius) * width;
    int shift = h * width;
    float *outPtr = output + shift; 
    int indexW = 0;

    float *tmpOutPtr = outPtr;
    float *tmpColSumPtr = colSumPtr;
    float *tmpAddPtr = addPtr;

    int nn = n;
    int remain = re;
#if __ARM_NEON
    for (; nn > 0; nn--) {
      float32x4_t _add = vld1q_f32(tmpAddPtr);
      float32x4_t _colSum = vld1q_f32(tmpColSumPtr);

      float32x4_t _tmp = vaddq_f32(_colSum, _add);

      vst1q_f32(tmpColSumPtr, _tmp);
      vst1q_f32(tmpOutPtr, _tmp);

      tmpAddPtr += 4;
      tmpColSumPtr += 4;
      tmpOutPtr += 4;
    }
#endif // __ARM_NEON
    for (; remain > 0; --remain) {
      *tmpColSumPtr += *tmpAddPtr;
      *tmpOutPtr = *tmpColSumPtr;
      tmpAddPtr ++;
      tmpColSumPtr ++;
      tmpOutPtr ++;
    }

  }

  int start = radius + 1;
  int end = height - 1 - radius;
  for (int h = start; h <= end; ++h) {
    float *addPtr = cachePtr + (h + radius) * width;
    float *subPtr = cachePtr + (h - radius - 1) * width;
    int shift = h * width;
    float *outPtr = output + shift; 
    int indexW = 0;
    float *tmpOutPtr = outPtr;
    float *tmpColSumPtr = colSumPtr;
    float *tmpAddPtr = addPtr;
    float *tmpSubPtr = subPtr;

    int nn = n;
    int remain = re;
#if __ARM_NEON
    for (; nn > 0; nn--) {
      float32x4_t _add = vld1q_f32(tmpAddPtr);
      float32x4_t _sub = vld1q_f32(tmpSubPtr);
      float32x4_t _colSum = vld1q_f32(tmpColSumPtr);

      float32x4_t _tmp = vaddq_f32(_colSum, _add);
      _tmp = vsubq_f32(_tmp, _sub);

      vst1q_f32(tmpColSumPtr, _tmp);
      vst1q_f32(tmpOutPtr, _tmp);

      tmpAddPtr += 4;
      tmpSubPtr += 4;
      tmpColSumPtr += 4;
      tmpOutPtr += 4;
    }
#endif // __ARM_NEON
    for (; remain > 0; --remain) {
      *tmpColSumPtr += *tmpAddPtr;
      *tmpColSumPtr -= *tmpSubPtr;
      *tmpOutPtr = *tmpColSumPtr;
      tmpAddPtr ++;
      tmpColSumPtr ++;
      tmpOutPtr ++;
      tmpSubPtr ++;
    }
  }

  start = height - radius;
  for (int h = start; h < height; ++h) {
    float *subPtr = cachePtr + (h - radius - 1) * width;
    int shift = h * width;
    float *outPtr = output + shift; 

    int indexW = 0;
    float *tmpOutPtr = outPtr;
    float *tmpColSumPtr = colSumPtr;
    float *tmpSubPtr = subPtr;

    int nn = n;
    int remain = re;
#if __ARM_NEON
    for (; nn > 0; nn--) {
      float32x4_t _sub = vld1q_f32(tmpSubPtr);
      float32x4_t _colSum = vld1q_f32(tmpColSumPtr);

      float32x4_t _tmp = vsubq_f32(_colSum, _sub);

      vst1q_f32(tmpColSumPtr, _tmp);
      vst1q_f32(tmpOutPtr, _tmp);

      tmpSubPtr += 4;
      tmpColSumPtr += 4;
      tmpOutPtr += 4;
    }
#endif // __ARM_NEON
    for (; remain > 0; --remain) {
      *tmpColSumPtr -= *tmpSubPtr;
      *tmpOutPtr = *tmpColSumPtr;
      tmpColSumPtr ++;
      tmpOutPtr ++;
      tmpSubPtr ++;
    }
  }
}

void BoxFilter::fastFilterV2NeonAsm(float *input, int radius, int height, int width, float *output) {
  float *cachePtr = &(cache[0]);
  // sum horizonal
  for (int h = 0; h < height; ++h) {
    int shift = h * width;

    float tmp = 0;
    for (int w = 0; w < radius; ++w) {
      tmp += input[shift + w];
    }

    for (int w = 0; w <= radius; ++w) {
      tmp += input[shift + w + radius];
      cachePtr[shift + w] = tmp;
    }

    int start = radius + 1;
    int end = width - 1 - radius;
    for (int w = start; w <= end; ++w) {
      tmp += input[shift + w + radius];
      tmp -= input[shift + w - radius - 1];
      cachePtr[shift + w] = tmp;
    }

    start = width - radius;
    for (int w = start; w < width; ++w) {
      tmp -= input[shift + w - radius - 1];
      cachePtr[shift + w] = tmp;
    }
  }

  float *colSumPtr = &(colSum[0]);

  for (int indexW = 0; indexW < width; ++indexW) {
    colSumPtr[indexW] = 0;
  } 

  int n = width >> 2;
  int re = width - (n << 2);
  // sum vertical
  for (int h = 0; h < radius; ++h) {
    int shift = h * width;

    float *tmpColSumPtr = colSumPtr;
    float *tmpCachePtr = cachePtr + shift;
    int indexW = 0;

    int nn = n;
    int remain = re;
#if __ARM_NEON
    for (; nn > 0; nn--) {
      float32x4_t _colSum = vld1q_f32(tmpColSumPtr);
      float32x4_t _cache = vld1q_f32(tmpCachePtr);

      float32x4_t _tmp = vaddq_f32(_colSum, _cache);

      vst1q_f32(tmpColSumPtr, _tmp); 

      tmpColSumPtr += 4;
      tmpCachePtr += 4;
    }
#endif // __ARM_NEON
    for (; remain > 0; --remain) {
      *tmpColSumPtr += *tmpCachePtr;
      tmpColSumPtr ++;
      tmpCachePtr ++;
    }
  }

  for (int h = 0; h <= radius; ++h) {
    float *addPtr = cachePtr + (h + radius) * width;
    int shift = h * width;
    float *outPtr = output + shift; 
    int indexW = 0;

    float *tmpOutPtr = outPtr;
    float *tmpColSumPtr = colSumPtr;
    float *tmpAddPtr = addPtr;

    int nn = n;
    int remain = re;
#if __ARM_NEON
    for (; nn > 0; nn--) {
      float32x4_t _add = vld1q_f32(tmpAddPtr);
      float32x4_t _colSum = vld1q_f32(tmpColSumPtr);

      float32x4_t _tmp = vaddq_f32(_colSum, _add);

      vst1q_f32(tmpColSumPtr, _tmp);
      vst1q_f32(tmpOutPtr, _tmp);

      tmpAddPtr += 4;
      tmpColSumPtr += 4;
      tmpOutPtr += 4;
    }
#endif // __ARM_NEON
    for (; remain > 0; --remain) {
      *tmpColSumPtr += *tmpAddPtr;
      *tmpOutPtr = *tmpColSumPtr;
      tmpAddPtr ++;
      tmpColSumPtr ++;
      tmpOutPtr ++;
    }

  }

  int start = radius + 1;
  int end = height - 1 - radius;
  for (int h = start; h <= end; ++h) {
    float *addPtr = cachePtr + (h + radius) * width;
    float *subPtr = cachePtr + (h - radius - 1) * width;
    int shift = h * width;
    float *outPtr = output + shift; 
    int indexW = 0;
    float *tmpOutPtr = outPtr;
    float *tmpColSumPtr = colSumPtr;
    float *tmpAddPtr = addPtr;
    float *tmpSubPtr = subPtr;

    int nn = n;
    int remain = re;
#if __ARM_NEON
    asm volatile(
      "0:                       \n"
      "vld1.s32 {d0-d1}, [%0]!  \n"
      "vld1.s32 {d2-d3}, [%1]!  \n"
      "vld1.s32 {d4-d5}, [%2]   \n"
      "vadd.f32 q4, q0, q2      \n"
      "vsub.f32 q3, q4, q1      \n"
      "vst1.s32 {d6-d7}, [%3]!  \n"
      "vst1.s32 {d6-d7}, [%2]!  \n"
      "subs %4, #1              \n"
      "bne  0b                  \n"
      : "=r"(tmpAddPtr), //
      "=r"(tmpSubPtr),
      "=r"(tmpColSumPtr),
      "=r"(tmpOutPtr),
      "=r"(nn)
      : "0"(tmpAddPtr),
      "1"(tmpSubPtr),
      "2"(tmpColSumPtr),
      "3"(tmpOutPtr),
      "4"(nn)
      : "cc", "memory", "q0", "q1", "q2", "q3", "q4"
    );

#endif // __ARM_NEON
    for (; remain > 0; --remain) {
      *tmpColSumPtr += *tmpAddPtr;
      *tmpColSumPtr -= *tmpSubPtr;
      *tmpOutPtr = *tmpColSumPtr;
      tmpAddPtr ++;
      tmpColSumPtr ++;
      tmpOutPtr ++;
      tmpSubPtr ++;
    }
  }

  start = height - radius;
  for (int h = start; h < height; ++h) {
    float *subPtr = cachePtr + (h - radius - 1) * width;
    int shift = h * width;
    float *outPtr = output + shift; 

    int indexW = 0;
    float *tmpOutPtr = outPtr;
    float *tmpColSumPtr = colSumPtr;
    float *tmpSubPtr = subPtr;

    int nn = n;
    int remain = re;
#if __ARM_NEON
    for (; nn > 0; nn--) {
      float32x4_t _sub = vld1q_f32(tmpSubPtr);
      float32x4_t _colSum = vld1q_f32(tmpColSumPtr);

      float32x4_t _tmp = vsubq_f32(_colSum, _sub);

      vst1q_f32(tmpColSumPtr, _tmp);
      vst1q_f32(tmpOutPtr, _tmp);

      tmpSubPtr += 4;
      tmpColSumPtr += 4;
      tmpOutPtr += 4;
    }
#endif // __ARM_NEON
    for (; remain > 0; --remain) {
      *tmpColSumPtr -= *tmpSubPtr;
      *tmpOutPtr = *tmpColSumPtr;
      tmpColSumPtr ++;
      tmpOutPtr ++;
      tmpSubPtr ++;
    }
  }
}

void BoxFilter::fastFilterV2NeonAsmV2(float *input, int radius, int height, int width, float *output) {
  float *cachePtr = &(cache[0]);
  // sum horizonal
  for (int h = 0; h < height; ++h) {
    int shift = h * width;

    float tmp = 0;
    for (int w = 0; w < radius; ++w) {
      tmp += input[shift + w];
    }

    for (int w = 0; w <= radius; ++w) {
      tmp += input[shift + w + radius];
      cachePtr[shift + w] = tmp;
    }

    int start = radius + 1;
    int end = width - 1 - radius;
    for (int w = start; w <= end; ++w) {
      tmp += input[shift + w + radius];
      tmp -= input[shift + w - radius - 1];
      cachePtr[shift + w] = tmp;
    }

    start = width - radius;
    for (int w = start; w < width; ++w) {
      tmp -= input[shift + w - radius - 1];
      cachePtr[shift + w] = tmp;
    }
  }

  float *colSumPtr = &(colSum[0]);

  for (int indexW = 0; indexW < width; ++indexW) {
    colSumPtr[indexW] = 0;
  } 

  // sum vertical
  for (int h = 0; h < radius; ++h) {
    int shift = h * width;

    float *tmpColSumPtr = colSumPtr;
    float *tmpCachePtr = cachePtr + shift;
    int indexW = 0;

    int nn = width >> 2;
    int remain = width - (nn << 2);
#if __ARM_NEON
    for (; nn > 0; nn--) {
      float32x4_t _colSum = vld1q_f32(tmpColSumPtr);
      float32x4_t _cache = vld1q_f32(tmpCachePtr);

      float32x4_t _tmp = vaddq_f32(_colSum, _cache);

      vst1q_f32(tmpColSumPtr, _tmp); 

      tmpColSumPtr += 4;
      tmpCachePtr += 4;
    }
#endif // __ARM_NEON
    for (; remain > 0; --remain) {
      *tmpColSumPtr += *tmpCachePtr;
      tmpColSumPtr ++;
      tmpCachePtr ++;
    }
  }

  for (int h = 0; h <= radius; ++h) {
    float *addPtr = cachePtr + (h + radius) * width;
    int shift = h * width;
    float *outPtr = output + shift; 
    int indexW = 0;

    float *tmpOutPtr = outPtr;
    float *tmpColSumPtr = colSumPtr;
    float *tmpAddPtr = addPtr;

    int nn = width >> 2;
    int remain = width - (nn << 2);
#if __ARM_NEON
    for (; nn > 0; nn--) {
      float32x4_t _add = vld1q_f32(tmpAddPtr);
      float32x4_t _colSum = vld1q_f32(tmpColSumPtr);

      float32x4_t _tmp = vaddq_f32(_colSum, _add);

      vst1q_f32(tmpColSumPtr, _tmp);
      vst1q_f32(tmpOutPtr, _tmp);

      tmpAddPtr += 4;
      tmpColSumPtr += 4;
      tmpOutPtr += 4;
    }
#endif // __ARM_NEON
    for (; remain > 0; --remain) {
      *tmpColSumPtr += *tmpAddPtr;
      *tmpOutPtr = *tmpColSumPtr;
      tmpAddPtr ++;
      tmpColSumPtr ++;
      tmpOutPtr ++;
    }

  }

  int start = radius + 1;
  int end = height - 1 - radius;
  for (int h = start; h <= end; ++h) {
    float *addPtr = cachePtr + (h + radius) * width;
    float *subPtr = cachePtr + (h - radius - 1) * width;
    int shift = h * width;
    float *outPtr = output + shift; 
    int indexW = 0;
    float *tmpOutPtr = outPtr;
    float *tmpColSumPtr = colSumPtr;
    float *tmpAddPtr = addPtr;
    float *tmpSubPtr = subPtr;

    int nn = width >> 3;
    int remain = width - (nn << 3);
#if __ARM_NEON
    asm volatile(
      "0:                       \n"
      "pld      [%0, #256]      \n"
      "vld1.s32 {d0-d3}, [%0]!  \n"
      "pld      [%2, #256]      \n"
      "vld1.s32 {d8-d11}, [%2]  \n"

      "vadd.f32 q6, q0, q4      \n"

      "pld      [%1, #256]      \n"
      "vld1.s32 {d4-d7}, [%1]!  \n"
      
      "vadd.f32 q7, q1, q5      \n"
      
      "vsub.f32 q6, q6, q2      \n"
      
      "vsub.f32 q7, q7, q3      \n"
      
      "vst1.s32 {d12-d15}, [%3]!  \n"
      
      "vst1.s32 {d12-d15}, [%2]!  \n"

      "subs %4, #1              \n"
      "bne  0b                  \n"
      : "=r"(tmpAddPtr), //
      "=r"(tmpSubPtr),
      "=r"(tmpColSumPtr),
      "=r"(tmpOutPtr),
      "=r"(nn)
      : "0"(tmpAddPtr),
      "1"(tmpSubPtr),
      "2"(tmpColSumPtr),
      "3"(tmpOutPtr),
      "4"(nn)
      : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9"
    );

#endif // __ARM_NEON
    for (; remain > 0; --remain) {
      *tmpColSumPtr += *tmpAddPtr;
      *tmpColSumPtr -= *tmpSubPtr;
      *tmpOutPtr = *tmpColSumPtr;
      tmpAddPtr ++;
      tmpColSumPtr ++;
      tmpOutPtr ++;
      tmpSubPtr ++;
    }
  }

  start = height - radius;
  for (int h = start; h < height; ++h) {
    float *subPtr = cachePtr + (h - radius - 1) * width;
    int shift = h * width;
    float *outPtr = output + shift; 

    int indexW = 0;
    float *tmpOutPtr = outPtr;
    float *tmpColSumPtr = colSumPtr;
    float *tmpSubPtr = subPtr;

    int nn = width >> 2;
    int remain = width - (nn << 2);
#if __ARM_NEON
    for (; nn > 0; nn--) {
      float32x4_t _sub = vld1q_f32(tmpSubPtr);
      float32x4_t _colSum = vld1q_f32(tmpColSumPtr);

      float32x4_t _tmp = vsubq_f32(_colSum, _sub);

      vst1q_f32(tmpColSumPtr, _tmp);
      vst1q_f32(tmpOutPtr, _tmp);

      tmpSubPtr += 4;
      tmpColSumPtr += 4;
      tmpOutPtr += 4;
    }
#endif // __ARM_NEON
    for (; remain > 0; --remain) {
      *tmpColSumPtr -= *tmpSubPtr;
      *tmpOutPtr = *tmpColSumPtr;
      tmpColSumPtr ++;
      tmpOutPtr ++;
      tmpSubPtr ++;
    }
  }
}

