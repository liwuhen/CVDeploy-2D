
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

static __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy){
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __global__ void decode_kernel(  // 每次处理一个框，每个线程并行处理。
    float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
    float* invert_affine_matrix, float* parray, int max_objects, int NUM_BOX_ELEMENT
) {  

    volatile int position = (blockDim.x * blockIdx.x + threadIdx.x);   // 1D线程索引
    if (position >= num_bboxes) return;

    // vscode调试控制台查看指针内容 *(@global float(*)[10])pitem表示查看前10个地址内容
    // 核函数地址，变量可以存储在寄存器中或local, shared, const 或者 global的内存
    // predict：85（5分别对应的是cx,cy,w,h,obj_conf，分别代表的含义是边界框中心点坐标、宽高、边界框内包含物体的置信度。80对应的是COCO数据中的80个类别。）
    float* pitem     = predict + (5 + num_classes) * position;  // pitem指向每个框的初始地址（每一行的首地址）
    float objectness = pitem[4];

    if(objectness < confidence_threshold)
        return;

    float* class_confidence = pitem + 5;  // classification 对应的首地址（第一个类别）
    float confidence        = *class_confidence++; // 第一个类别地址对应的预测的类别概率值
    int label               = 0;
    for(int i = 1; i < num_classes; ++i, ++class_confidence) { // 查找预测的80类别概率值中最大的类
        if(*class_confidence > confidence) {
            confidence = *class_confidence;
            label      = i;
        }
    }

    confidence *= objectness;  // 对应类别置信度值（表示边界框内存在物体且物体是这个类别的概率）
    if(confidence < confidence_threshold)
        return;

    // volatile 编译器对访问该变量的代码就不再进行优化，从而可以提供对特殊地址的稳定访问
    // 这里的parray是一段连续的内存空间数组， atomicAdd(parray, 1)的功能是parray中每个元素都同时加1，并返回old_value。
    int index = 0;
    index = atomicAdd(parray, 1);  // 计算(old + val)， 函数将返回old 地址对应的新值; 所有线程在这里会阻塞，在共享内存parray中进行计算。
    
    if(index >= max_objects)
        return;

    float cx         = *pitem++;
    float cy         = *pitem++;
    float width      = *pitem++;
    float height     = *pitem++;
    float left   = cx - width * 0.5f;
    float top    = cy - height * 0.5f;
    float right  = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    // affine_project(invert_affine_matrix, left,  top,    &left,  &top);
    // affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    // left, top, right, bottom, confidence, class, keepflag
    // 这里的+1表示存放的bbox的数量count的信息。
    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;   // pout_item是output_device的每一个box信息的首地址。存在parray对应的内存地址空间中。
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; // 1 = keep, 0 = ignore

    // cuda-gdb调试变量显示optimized_out 
    printf("index=%d, num_bboxes=%d, position=%d, threadIdx.x=%d, blockIdx.x=%d, blockDim.x=%d\n", index, num_bboxes, position, threadIdx.x, blockIdx.x, blockDim.x); 
}

static __device__ float box_iou(
    float aleft, float atop, float aright, float abottom, 
    float bleft, float btop, float bright, float bbottom
){

    float cleft 	= max(aleft, bleft);
    float ctop 		= max(atop, btop);
    float cright 	= min(aright, bright);
    float cbottom 	= min(abottom, bbottom);
    
    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if(c_area == 0.0f)
        return 0.0f;
    
    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void fast_nms_kernel(float* bboxes, int max_objects, float threshold, int NUM_BOX_ELEMENT){

    // 如果测mAP的性能的时候， 只能采用cpu nms
    // 如果是日常推理， 则可以使用这个gpu nms
    int position = (blockDim.x * blockIdx.x + threadIdx.x);  
    int count    = min((int)*bboxes, max_objects);  // *bboxes表示首地址的第一个元素。 count是bbox的数量。
    if (position >= count) 
        return; 
    
    // left, top, right, bottom, confidence, class, keepflag
    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT; // +1是因为bboxes中第一个元素是记录bbox的数量的标识。
    for(int i = 0; i < count; ++i) {
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        if(i == position || pcurrent[5] != pitem[5]) continue;  // 剔除自己与类别不一致的框

        if(pitem[4] >= pcurrent[4]) {
            if(pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou = box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0],    pitem[1],    pitem[2],    pitem[3]
            );

            if(iou > threshold){
                pcurrent[6] = 0;  // 1=keep, 0=ignore
                return;
            }
        }
    }
} 

void decode_kernel_invoker(
    float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
    float nms_threshold, float* invert_affine_matrix, float* parray, int max_objects, int NUM_BOX_ELEMENT, cudaStream_t stream){
    
    auto block = num_bboxes > 512 ? 512 : num_bboxes; // num_bboxes的数量框，则需要num_bboxes的线程数（一线程处理一个框）。
    auto grid  = (num_bboxes + block - 1) / block;    // +block -1是向上取整数，保证系统设定的线程数大于程序的需要。 线程块的数量应该满足num_bboxes所需的线程数。

    /* 如果核函数有波浪线，没关系，他是正常的，你只是看不顺眼罢了 */
    decode_kernel<<<grid, block, 0, stream>>>(
        predict, num_bboxes, num_classes, confidence_threshold, 
        invert_affine_matrix, parray, max_objects, NUM_BOX_ELEMENT
    );

    block = max_objects > 512 ? 512 : max_objects;  // 每一个图像最多的框数量:max_objects
    grid = (max_objects + block - 1) / block;
    fast_nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold, NUM_BOX_ELEMENT); //parray表示解码后的bbox
}