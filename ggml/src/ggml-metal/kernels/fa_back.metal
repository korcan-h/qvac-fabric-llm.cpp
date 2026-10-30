#include "common.h"

// o and grad_o are [D, n_head, n_q, ne3], stats [2, n_q, n_head, ne3], contiguous like dst:
//   STATS: dst [2, n_q, n_head, ne3], (logsumexp, dot(grad_o, o)) per q row
//   DQ:    dst [D, n_q, n_head, ne3]
//   DKV:   dst [D, n_kv, n_head_kv, 2*ne3], dk in ne3 [0, ne3), dv in [ne3, 2*ne3)

template <typename T, const uint FAB_PER_LANE>
inline void fab_load_row(device const T * row,
                         const int64_t D,
                         const uint lane,
                         thread float (&x)[FAB_PER_LANE]) {
    FOR_UNROLL (int i = 0; i < FAB_PER_LANE; ++i) {
        const uint d = lane + i * N_SIMDWIDTH;
        x[i] = d < D ? float(row[d]) : 0.0f;
    }
}

template <const uint FAB_PER_LANE>
inline void fab_store_row(device float * row,
                          const int64_t D,
                          const uint lane,
                          thread float (&x)[FAB_PER_LANE]) {
    FOR_UNROLL(int i = 0; i < FAB_PER_LANE; ++i) {
        const int d = lane + i * N_SIMDWIDTH;
        if (d < D) {
            row[d] = x[i];
        }
    }
}

template <const uint FAB_PER_LANE>
inline float fab_dot(thread float (&x)[FAB_PER_LANE],
                     thread float (&y)[FAB_PER_LANE]) {
    float s = 0.0f;
    FOR_UNROLL (int i = 0; i < FAB_PER_LANE; ++i) {
        s = fma(x[i], y[i], s);
    }
    return s;
}

// mask value of q row (i1, i2, i3) at kv position ic, the same for every lane of the warp
// mask is nullptr when absent, the host binds q in its slot and the kernels check for that
inline float fab_mask(constant ggml_metal_kargs_ext_back_kargs & args,
                      device const half * mask,
                      const int64_t i1, const int64_t i2,
                      const int64_t i3, const int64_t ic) {
    if (!mask) {
        return 0.0f;
    }
    device const char * row = (device const char *) mask + i1*args.nbm1 + (i2 % args.nem2)*args.nbm2 + (i3 % args.nem3)*args.nbm3;
    return float(((device const half *) row)[ic]);
}

template <typename T>
inline device const T * fab_row(device const T * base, const uint64_t offs) {
    return (device const T *) ((device const char *) base + offs);
}

// row of o and grad_o for q row (i1, i2, i3), heads are on dim 1 there
inline int64_t fab_o_row(constant ggml_metal_kargs_ext_back_kargs & args,
                         const int64_t i1, const int64_t i2, const int64_t i3) {
    return (i2 + i1*args.n_head + i3*args.n_head*args.n_q)*args.D;
}

template <typename T_KV, uint D_MAX, uint NSG>
kernel void flash_attn_ext_back_stats(
        constant ggml_metal_kargs_ext_back_kargs & args [[buffer(0)]],
        device const float * q      [[buffer(1)]],
        device const T_KV  * k      [[buffer(2)]],
        device const half  * mask   [[buffer(5)]],
        device const float * o      [[buffer(6)]],
        device const float * grad_o [[buffer(3)]],
              device float * dst    [[buffer(4)]],
                     uint    tgpig  [[threadgroup_position_in_grid]],
                     ushort  sgitg  [[simdgroup_index_in_threadgroup]],
                     ushort  tiisg  [[thread_index_in_simdgroup]]) {

    constexpr uint FAB_PER_LANE = D_MAX / N_SIMDWIDTH;
    device const half * pmask = (device const char *) mask != (device const char *) q ? mask : nullptr;

    const int64_t ir = (int64_t) tgpig*NSG + sgitg;
    if (ir >= args.n_q*args.n_head*args.ne3) {
        return;
    }

    const int64_t i1  = ir % args.n_q;
    const int64_t i2  = (ir/args.n_q) % args.n_head;
    const int64_t i3  = ir/(args.n_q*args.n_head);
    const int64_t ik2 = i2/(args.n_head/args.n_head_kv);

    float x[FAB_PER_LANE];
    float y[FAB_PER_LANE];
    fab_load_row(fab_row(q, i1*args.nbq1 + i2*args.nbq2 + i3*args.nbq3), args.D, tiisg, x);

    float S = 0.0f;
    float M = -FLT_MAX/2;
    for (int64_t ic = 0; ic < args.n_kv; ++ic) {
        const float mv = fab_mask(args, pmask, i1, i2, i3, ic);
        if (mv <= -MAXHALF) {
            continue;
        }
        fab_load_row(fab_row(k, ic*args.nbk1 + ik2*args.nbk2 + i3*args.nbk3), args.D, tiisg, y);
        const float s = simd_sum(fab_dot(x, y))*args.scale + mv;
        if (s > M) {
            S = S*exp(M - s) + 1.0f;
            M = s;
        } else {
            S += exp(s - M);
        }
    }

    const int64_t io = fab_o_row(args, i1, i2, i3);
    fab_load_row(o      + io, args.D, tiisg, x);
    fab_load_row(grad_o + io, args.D, tiisg, y);
    const float d = simd_sum(fab_dot(x, y));

    if (tiisg == 0) {
        dst[2*ir + 0] = S == 0.0f ? -INFINITY : M + log(S);
        dst[2*ir + 1] = d;
    }
}

template <typename T_KV, uint D_MAX, uint NSG>
kernel void flash_attn_ext_back_dq(
    constant ggml_metal_kargs_ext_back_kargs & args [[buffer(0)]],
    device const float * q      [[buffer(1)]],
    device const T_KV  * k      [[buffer(2)]],
    device const T_KV  * v      [[buffer(7)]],
    device const half  * mask   [[buffer(5)]],
    device const float * grad_o [[buffer(3)]],
    device const float * stats  [[buffer(8)]],
          device float * dst    [[buffer(4)]],
                 uint    tgpig  [[threadgroup_position_in_grid]],
                 ushort  sgitg  [[simdgroup_index_in_threadgroup]],
                 ushort  tiisg  [[thread_index_in_simdgroup]]) {

    constexpr uint FAB_PER_LANE = D_MAX / N_SIMDWIDTH;
    device const half * pmask = (device const char *) mask != (device const char *) q ? mask : nullptr;

    const int64_t ir = (int64_t) tgpig*NSG + sgitg;
    if (ir >= args.n_q*args.n_head*args.ne3) {
        return;
    }
    const int64_t i1  = ir % args.n_q;
    const int64_t i2  = (ir/args.n_q) % args.n_head;
    const int64_t i3  = ir/(args.n_q*args.n_head);
    const int64_t ik2 = i2/(args.n_head/args.n_head_kv);

    float x[FAB_PER_LANE];
    float g[FAB_PER_LANE];
    float y[FAB_PER_LANE];
    float z[FAB_PER_LANE];
    float acc[FAB_PER_LANE] = {0.0f};
    fab_load_row(fab_row(q, i1*args.nbq1 + i2*args.nbq2 + i3*args.nbq3), args.D, tiisg, x);
    fab_load_row(grad_o + fab_o_row(args, i1, i2, i3), args.D, tiisg, g);

    const float L  = stats[2*ir + 0];
    const float Do = stats[2*ir + 1];

    for (int64_t ic = 0; ic < args.n_kv; ++ic) {
        const float mv = fab_mask(args, pmask, i1, i2, i3, ic);
        if (mv <= -MAXHALF) {
            continue;
        }
        fab_load_row(fab_row(k, ic*args.nbk1 + ik2*args.nbk2 + i3*args.nbk3), args.D, tiisg, y);
        fab_load_row(fab_row(v, ic*args.nbv1 + ik2*args.nbv2 + i3*args.nbv3), args.D, tiisg, z);
        const float2 sd = simd_sum(float2(fab_dot(x, y), fab_dot(g, z)));
        const float p  = exp(sd.x*args.scale + mv - L);
        const float ds = p*(sd.y - Do);
        FOR_UNROLL (int i = 0; i < FAB_PER_LANE; ++i) {
            acc[i] += args.scale*ds*y[i];
        }
    }

    fab_store_row(dst + ir*args.D, args.D, tiisg, acc);
}

template < typename T_KV, uint D_MAX, uint NSG >
kernel void flash_attn_ext_back_dkv(
        constant ggml_metal_kargs_ext_back_kargs& args [[buffer(0)]],
        device const float * q      [[buffer(1)]],
        device const T_KV  * k      [[buffer(2)]],
        device const T_KV  * v      [[buffer(7)]],
        device const half  * mask   [[buffer(5)]],
        device const float * grad_o [[buffer(3)]],
        device const float * stats  [[buffer(8)]],
              device float * dst    [[buffer(4)]],
                     uint    tgpig  [[threadgroup_position_in_grid]],
                     ushort  sgitg  [[simdgroup_index_in_threadgroup]],
                     ushort  tiisg  [[thread_index_in_simdgroup]]) {

    constexpr uint FAB_PER_LANE = D_MAX / N_SIMDWIDTH;
    device const half * pmask = (device const char *) mask != (device const char *) q ? mask : nullptr;

    const int64_t ir     = (int64_t) tgpig*NSG + sgitg;
    const int64_t n_rows = args.n_kv*args.n_head_kv*args.ne3;
    if (ir >= n_rows) {
        return;
    }
    const int64_t ik1 = ir % args.n_kv;
    const int64_t ik2 = (ir/args.n_kv) % args.n_head_kv;
    const int64_t ik3 = ir/(args.n_kv*args.n_head_kv);
    const int64_t rk2 = args.n_head/args.n_head_kv;

    float x[FAB_PER_LANE];
    float g[FAB_PER_LANE];
    float y[FAB_PER_LANE];
    float z[FAB_PER_LANE];
    float dk[FAB_PER_LANE] = {0.0f};
    float dv[FAB_PER_LANE] = {0.0f};
    fab_load_row(fab_row(k, ik1*args.nbk1 + ik2*args.nbk2 + ik3*args.nbk3), args.D, tiisg, y);
    fab_load_row(fab_row(v, ik1*args.nbv1 + ik2*args.nbv2 + ik3*args.nbv3), args.D, tiisg, z);

    // all q heads of this kv head (GQA reduce)
    for (int64_t iq2 = ik2*rk2; iq2 < (ik2 + 1)*rk2; ++iq2) {
        for (int64_t iq1 = 0; iq1 < args.n_q; ++iq1) {
            const float mv = fab_mask(args, pmask, iq1, iq2, ik3, ik1);
            if (mv <= -MAXHALF) {
                continue;
            }
            fab_load_row(fab_row(q, iq1*args.nbq1 + iq2*args.nbq2 + ik3*args.nbq3), args.D, tiisg, x);
            fab_load_row(grad_o + fab_o_row(args, iq1, iq2, ik3), args.D, tiisg, g);
            device const float * st = stats + 2*(iq1 + iq2*args.n_q + ik3*args.n_q*args.n_head);

            const float2 sd = simd_sum(float2(fab_dot(x, y), fab_dot(g, z)));
            const float p  = exp(sd.x*args.scale + mv - st[0]);
            const float ds = p*(sd.y - st[1]);
            FOR_UNROLL (int i = 0; i < FAB_PER_LANE; ++i) {
                dv[i] += p*g[i];
                dk[i] += args.scale*ds*x[i];
            }
        }
    }

    fab_store_row(dst +  ir          *args.D, args.D, tiisg, dk);
    fab_store_row(dst + (ir + n_rows)*args.D, args.D, tiisg, dv);
}

#define FAB_NSG 8

typedef decltype(flash_attn_ext_back_stats<float, 64, FAB_NSG>) flash_attn_ext_back_stats_f32_t;
typedef decltype(flash_attn_ext_back_stats<half,  64, FAB_NSG>) flash_attn_ext_back_stats_f16_t;
typedef decltype(flash_attn_ext_back_dq   <float, 64, FAB_NSG>) flash_attn_ext_back_dq_f32_t;
typedef decltype(flash_attn_ext_back_dq   <half,  64, FAB_NSG>) flash_attn_ext_back_dq_f16_t;
typedef decltype(flash_attn_ext_back_dkv  <float, 64, FAB_NSG>) flash_attn_ext_back_dkv_f32_t;
typedef decltype(flash_attn_ext_back_dkv  <half,  64, FAB_NSG>) flash_attn_ext_back_dkv_f16_t;

template [[host_name("kernel_flash_attn_ext_back_stats_f32_d64")]]  kernel flash_attn_ext_back_stats_f32_t flash_attn_ext_back_stats<float,  64, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_stats_f32_d128")]] kernel flash_attn_ext_back_stats_f32_t flash_attn_ext_back_stats<float, 128, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_stats_f32_d256")]] kernel flash_attn_ext_back_stats_f32_t flash_attn_ext_back_stats<float, 256, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_stats_f16_d64")]]  kernel flash_attn_ext_back_stats_f16_t flash_attn_ext_back_stats<half,   64, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_stats_f16_d128")]] kernel flash_attn_ext_back_stats_f16_t flash_attn_ext_back_stats<half,  128, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_stats_f16_d256")]] kernel flash_attn_ext_back_stats_f16_t flash_attn_ext_back_stats<half,  256, FAB_NSG>;

template [[host_name("kernel_flash_attn_ext_back_dq_f32_d64")]]     kernel flash_attn_ext_back_dq_f32_t    flash_attn_ext_back_dq   <float,  64, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_dq_f32_d128")]]    kernel flash_attn_ext_back_dq_f32_t    flash_attn_ext_back_dq   <float, 128, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_dq_f32_d256")]]    kernel flash_attn_ext_back_dq_f32_t    flash_attn_ext_back_dq   <float, 256, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_dq_f16_d64")]]     kernel flash_attn_ext_back_dq_f16_t    flash_attn_ext_back_dq   <half,   64, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_dq_f16_d128")]]    kernel flash_attn_ext_back_dq_f16_t    flash_attn_ext_back_dq   <half,  128, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_dq_f16_d256")]]    kernel flash_attn_ext_back_dq_f16_t    flash_attn_ext_back_dq   <half,  256, FAB_NSG>;

template [[host_name("kernel_flash_attn_ext_back_dkv_f32_d64")]]    kernel flash_attn_ext_back_dkv_f32_t   flash_attn_ext_back_dkv  <float,  64, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_dkv_f32_d128")]]   kernel flash_attn_ext_back_dkv_f32_t   flash_attn_ext_back_dkv  <float, 128, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_dkv_f32_d256")]]   kernel flash_attn_ext_back_dkv_f32_t   flash_attn_ext_back_dkv  <float, 256, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_dkv_f16_d64")]]    kernel flash_attn_ext_back_dkv_f16_t   flash_attn_ext_back_dkv  <half,   64, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_dkv_f16_d128")]]   kernel flash_attn_ext_back_dkv_f16_t   flash_attn_ext_back_dkv  <half,  128, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_dkv_f16_d256")]]   kernel flash_attn_ext_back_dkv_f16_t   flash_attn_ext_back_dkv  <half,  256, FAB_NSG>;
