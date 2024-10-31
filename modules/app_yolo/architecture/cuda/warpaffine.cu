#include "warpaffine.hpp"

__device__ void affine_project(Matrix3f affineMatrix_inv, int x, int y, float* proj_x, float* proj_y){

    *proj_x = affineMatrix_inv(0,0) * x + affineMatrix_inv(0,1) * y + affineMatrix_inv(0,2);
    *proj_y = affineMatrix_inv(1,0) * x + affineMatrix_inv(1,1) * y + affineMatrix_inv(1,2);
}

static __global__ void warp_affine_bilinear_kernel(
    uint8_t* src, int batch, int src_line_size, int src_width, int src_height, 
    float* dst, int dst_width, int dst_height, Matrix3f affineMatrix_inv, 
    uint8_t const_value_st, AppYolo app_mode)
{
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    int k  = blockDim.z * blockIdx.z + threadIdx.z;

    if (dx >= dst_width || dy >= dst_height || k >= batch)
        return;

    uint8_t *srcs = src + k * src_width * src_height * 3;
    float   *dsts = dst + k * dst_width * dst_height * 3;

    float src_x = 0; float src_y = 0;
    affine_project(affineMatrix_inv, dx, dy, &src_x, &src_y); 

    float c0 = const_value_st, c1 = const_value_st, c2 = const_value_st;

    if(src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height){
        // out of range
        // c0 = const_value_st;
        // c1 = const_value_st;
        // c2 = const_value_st;
    }else{
        int y_low  = floorf(src_y);
        int x_low  = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
        float ly    = src_y - y_low;
        float lx    = src_x - x_low;
        float hy    = 1 - ly;
        float hx    = 1 - lx;
        float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t* v1 = const_value;
        uint8_t* v2 = const_value;
        uint8_t* v3 = const_value;
        uint8_t* v4 = const_value;
        if(y_low >= 0){
            if (x_low >= 0)
                v1 = srcs + y_low * src_line_size + x_low * 3;

            if (x_high < src_width)
                v2 = srcs + y_low * src_line_size + x_high * 3;
        }
        
        if(y_high < src_height){
            if (x_low >= 0)
                v3 = srcs + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = srcs + y_high * src_line_size + x_high * 3;
        }

        c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);   // b
        c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);   // g
        c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);   // r
    }


    if ( app_mode == AppYolo::YOLOV5_MODE ) {
        // normalisation
        c0 = (c0 / 255.0f);
        c1 = (c1 / 255.0f);
        c2 = (c2 / 255.0f);

        // bgr to rgb
        int area = dst_width * dst_height;
        float* pdst_b = dsts + area * 0 + dy * dst_width + dx;
        float* pdst_g = dsts + area * 1 + dy * dst_width + dx;
        float* pdst_r = dsts + area * 2 + dy * dst_width + dx;
        *pdst_r = c0;
        *pdst_g = c1;
        *pdst_b = c2;
    } else if ( app_mode == AppYolo::YOLOX_MODE )
    {
        int area = dst_width * dst_height;
        float* pdst_b = dsts + area * 0 + dy * dst_width + dx;
        float* pdst_g = dsts + area * 1 + dy * dst_width + dx;
        float* pdst_r = dsts + area * 2 + dy * dst_width + dx;
        *pdst_r = c2;
        *pdst_g = c1;
        *pdst_b = c0;
    }
}

void warp_affine_bilinear(
    uint8_t* src, int batch, InfertMsg& input_msg, float* dst, int dst_width, int dst_height,
    uint8_t const_value, cudaStream_t stream, AppYolo app_mode) 
{

    dim3 block_size(16, 16, 4); // blocksize最大就是1024
    dim3 grid_size((dst_width + 15) / 16, (dst_height + 15) / 16, (batch + 3) / 4);

    warp_affine_bilinear_kernel <<<grid_size, block_size, 0, stream>>> (
        src, batch, input_msg.width * 3, input_msg.width, input_msg.height, 
        dst, dst_width, dst_height, input_msg.affineMatrix_inv, const_value, app_mode
    );
}
