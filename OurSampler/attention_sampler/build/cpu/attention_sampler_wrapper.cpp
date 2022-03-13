/*
 * MobulaOP Wrapper generated from the source code attention_sampler/attention_sampler.cpp
 * Created by: MobulaOP ae693a6
 * Create Time: Thu 2021-07-08 13:48:51 (+0800)
 *
 * WARNING! All changes made in this file will be lost!
 */
#include "mobula_op.h"
using namespace mobula;

#include "/home/yaoshuilian/TASN/C2FNet-master2/demo/attention_sampler/attention_sampler.cpp"

extern "C" {

MOBULA_DLL void map_step_6174d786(const int device_id, const int N, const float* attxi, float* index_x, const float* stepxs, const int att_size, const int out_size) {
  KERNEL_RUN_BEGIN(device_id);
  KERNEL_RUN((map_step_kernel<float>))(N, attxi, index_x, stepxs, att_size, out_size);
  KERNEL_RUN_END();
}

MOBULA_DLL PackedFunc* map_step_6174d786_register_mx() {
  return GetMXNetFunc(
      "map_step_6174d786",
      [](TVMArgs args, TVMRetValue*) {
        KERNEL_RUN_BEGIN(DEV_ID);
        KERNEL_RUN_STREAM((map_step_kernel<float>), STRM)(
          args.values[0].v_int64,
          static_cast<const float*>(
            static_cast<DLTensor*>(args.values[1].v_handle)->data),
          static_cast<float*>(
            static_cast<DLTensor*>(args.values[2].v_handle)->data),
          static_cast<const float*>(
            static_cast<DLTensor*>(args.values[3].v_handle)->data),
          args.values[4].v_int64,
          args.values[5].v_int64
        );
        KERNEL_RUN_END();
      },
      2, std::array<int, 2>({1,3}).data());
}

MOBULA_DLL void map_step_6174d786_async_mx(
    PackedFunc* packed_func,
    const int N, NDArrayHandle attxi, NDArrayHandle index_x, NDArrayHandle stepxs, const int att_size, const int out_size) {
  (*packed_func)(N, attxi, index_x, stepxs, att_size, out_size);
}


}
