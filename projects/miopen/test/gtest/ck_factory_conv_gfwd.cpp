// Unit tests using gtest for the CK grouped convolution forward factory.
//
// ## Purpose:
// Checks that the CK factory for grouped convolution forward operations (with different
// epilogue types: PassThrough, Scale, Bilinear) creates the expected instances for the
// tested data types (fp16, fp32, bf16, s8).
//
// This test is very vebose because it is using to guard against regressions as we migrate
// to a better solution of instantiating these instances directly in MIOpen.
//
// ## How:
// Each test calls GetInstances() on a factory type for a specific combination of data type and
// epilogue and asserts that the returned vector of instances matches the values that are returned
// currently by CK.
//
// ## Why:
// This verifies that the instance factory is producing consistent output. If we change the
// available kernels we should update this test.
//
// ## Tested Types:
// - Data types:
//   - fp16: ck::half_t
//   - fp32: float
//   - bf16: ck::bhalf_t
//   - s8: int8_t
// - Epilogues: PassThrough, Scale, Bilinear
#include <miopen/config.h>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <gtest/gtest.h>
#include <array>
#include <string_view>

#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bilinear.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scale.hpp>

namespace miopen::unit_test {
namespace {

using InLayout                             = ck::tensor_layout::convolution::NDHWGC;
using WeiLayout                            = ck::tensor_layout::convolution::GKZYXC;
using OutLayout                            = ck::tensor_layout::convolution::NDHWGK;
using PassThrough                          = ck::tensor_operation::element_wise::PassThrough;
using Bilinear                             = ck::tensor_operation::element_wise::Bilinear;
using Scale                                = ck::tensor_operation::element_wise::Scale;
static constexpr ck::index_t NumDimSpatial = 3;

template <typename DataType, typename Epilogue>
struct DeviceOpGFwdHelper
{
    using type = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                                               InLayout,
                                                                               WeiLayout,
                                                                               ck::Tuple<>,
                                                                               OutLayout,
                                                                               DataType,
                                                                               DataType,
                                                                               ck::Tuple<>,
                                                                               DataType,
                                                                               PassThrough,
                                                                               PassThrough,
                                                                               Epilogue>;
};

template <typename DataType>
struct DeviceOpGFwdHelper<DataType, Bilinear>
{
    // Bilinear specialization has OutLayout and DataType.
    using type = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                                               InLayout,
                                                                               WeiLayout,
                                                                               ck::Tuple<OutLayout>,
                                                                               OutLayout,
                                                                               DataType,
                                                                               DataType,
                                                                               ck::Tuple<DataType>,
                                                                               DataType,
                                                                               PassThrough,
                                                                               PassThrough,
                                                                               Bilinear>;
};

template <typename DataType, typename Epilogue>
using DeviceOpGFwdPtrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
    typename DeviceOpGFwdHelper<DataType, Epilogue>::type>;

template <typename T>
class CkFactoryTest : public testing::Test
{
};

struct Fp16PassThrough
{
    using DataType                                                    = ck::half_t;
    using Epilogue                                                    = PassThrough;
    static constexpr int COUNT                                        = 228;
    static constexpr std::array<std::string_view, COUNT> TYPE_STRINGS = {
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Default, 32, 32, 2, 2, 1, 1, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Default, 32, 32, 2, 1, 8, 8, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Default, 32, 32, 2, 2, 1, "
        "1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Default, 32, 32, 4, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Default, 32, 32, 2, 4, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Default, 32, 32, 4, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Default, 32, 32, 2, 2, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Default, 32, 32, 2, 1, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Default, 32, 32, 1, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Default, 32, 32, 2, 1, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Default, 32, 32, 1, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Default, 32, 32, 2, 1, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Default, 32, 32, 1, 2, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Pad0, 32, 32, 2, 2, "
        "1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Pad0, 32, 32, 2, 1, "
        "8, 8, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 1, 1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Filter1x1Pad0, 32, 32, 4, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Filter1x1Pad0, 32, 32, 2, "
        "4, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Filter1x1Pad0, 32, 32, 4, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Pad0, 32, 32, 2, 2, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Filter1x1Pad0, 32, 32, 2, "
        "1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Filter1x1Pad0, 32, 32, 1, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Filter1x1Pad0, 32, 32, 2, "
        "1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Filter1x1Pad0, 32, 32, 1, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Pad0, 32, 32, 2, 1, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Filter1x1Pad0, 32, 32, 1, 2, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 4, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Default, 16, 16, 2, 2, 1, "
        "2, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Default, 16, 16, 2, 2, 2, "
        "1, 2, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Default, 16, 16, 2, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Default, 16, 16, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Pad0, 16, 16, 2, "
        "2, 1, 2, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Pad0, 16, 16, 2, "
        "2, 2, 1, 2, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Pad0, 16, 16, 2, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Pad0, 16, 16, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Stride1Pad0, 16, "
        "16, 2, 2, 1, 2, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Stride1Pad0, 16, "
        "16, 2, 2, 2, 1, 2, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Stride1Pad0, 16, "
        "16, 2, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Stride1Pad0, 16, "
        "16, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<64, 64, 64, 32, Default, 32, 32, "
        "2, 2, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<256, 256, 128, 32, Default, 32, "
        "32, 4, 2, 2, 2, 2, 1, 1>",
        "DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<256, 256, 128, 32, Default, 32, "
        "32, 4, 2, 8, 8, 8, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 16, Default, 16, 16, 4, 1, 4, 1, "
        "1, 1, 1, 8>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 16, Default, 16, 16, 4, 1, 4, 1, "
        "1, 1, 1, 16>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 16, Default, 16, 16, 4, 1, 4, 1, "
        "1, 1, 1, 32>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 16, Filter3x3, 16, 16, 4, 1, 4, "
        "1, 1, 1, 1, 8>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 16, Filter3x3, 16, 16, 4, 1, 4, "
        "1, 1, 1, 1, 16>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 16, Filter3x3, 16, 16, 4, 1, 4, "
        "1, 1, 1, 1, 32>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Default, 32, 32, 2, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Filter1x1Pad0, 32, 32, "
        "2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Filter1x1Stride1Pad0, "
        "32, 32, 2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v4>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Default, 32, 32, 4, 4, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Filter1x1Pad0, 32, 32, "
        "4, 4, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Filter1x1Stride1Pad0, "
        "32, 32, 4, 4, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v4>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 32, Default, 32, 32, 2, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 32, Filter1x1Pad0, 32, 32, "
        "2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 32, Filter1x1Stride1Pad0, "
        "32, 32, 2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v4>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Default, 32, 32, 4, 4, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Filter1x1Pad0, 32, 32, "
        "4, 4, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Filter1x1Stride1Pad0, "
        "32, 32, 4, 4, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Default, 32, 32, 4, 4, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v5>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Filter1x1Pad0, 32, 32, "
        "4, 4, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v5>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Filter1x1Stride1Pad0, "
        "32, 32, 4, 4, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v5>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Default, 16, 16, 8, 8, "
        "8, 8, 8, 1, 2, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Filter1x1Pad0, 16, 16, "
        "8, 8, 8, 8, 8, 1, 2, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Filter1x1Stride1Pad0, "
        "16, 16, 8, 8, 8, 8, 8, 1, 2, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 224, 256, 64, Default, 16, 16, 7, 8, "
        "8, 8, 8, 1, 2, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 224, 256, 64, Filter1x1Pad0, 16, 16, "
        "7, 8, 8, 8, 8, 1, 2, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 224, 256, 64, Filter1x1Stride1Pad0, "
        "16, 16, 7, 8, 8, 8, 8, 1, 2, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 224, 64, Default, 16, 16, 8, 7, "
        "8, 8, 8, 2, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 224, 64, Filter1x1Pad0, 16, 16, "
        "8, 7, 8, 8, 8, 2, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 224, 64, Filter1x1Stride1Pad0, "
        "16, 16, 8, 7, 8, 8, 8, 2, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Default, 32, 32, 2, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Filter1x1Pad0, 32, 32, "
        "2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Filter1x1Stride1Pad0, "
        "32, 32, 2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Default, 32, 32, 2, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v5>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Filter1x1Pad0, 32, 32, "
        "2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v5>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Filter1x1Stride1Pad0, "
        "32, 32, 2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v5>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 256, 32, Default, 32, 32, 2, 4, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 256, 32, Filter1x1Pad0, 32, 32, "
        "2, 4, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 256, 32, Filter1x1Stride1Pad0, "
        "32, 32, 2, 4, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 128, 32, Default, 32, 32, 4, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 128, 32, Filter1x1Pad0, 32, 32, "
        "4, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 128, 32, Filter1x1Stride1Pad0, "
        "32, 32, 4, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Default, 32, 32, 2, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Filter1x1Pad0, 32, 32, "
        "2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Filter1x1Stride1Pad0, "
        "32, 32, 2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Default, 16, 16, 1, 1, "
        "8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Default, 16, 16, 1, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Default, 16, 16, 1, 1, 8, "
        "8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Default, 16, 16, 1, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 32, 64, Default, 32, 32, 2, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 16, 64, Default, 16, 16, 4, 1, "
        "8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 32, 64, Default, 32, 32, 2, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 16, 64, Default, 16, 16, 4, 1, "
        "8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 32, 64, Default, 32, 32, 1, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 16, 64, Default, 16, 16, 2, 1, "
        "8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Default, 16, 16, 1, 1, "
        "8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Default, 16, 16, 1, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Default, 16, 16, 1, 1, 8, "
        "8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Default, 16, 16, 1, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 64, 64, Default, 16, 16, 1, 2, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 64, 64, Default, 32, 32, 1, 1, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 128, 64, Default, 16, 16, 1, 4, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 128, 64, Default, 32, 32, 1, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 16, 256, 64, Default, 16, 16, 1, 4, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 32, 256, 64, Default, 32, 32, 1, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Pad0, 16, 16, 1, "
        "1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 32, 64, Filter1x1Pad0, 32, 32, "
        "2, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 16, 64, Filter1x1Pad0, 16, 16, "
        "4, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 32, 64, Filter1x1Pad0, 32, 32, "
        "2, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 16, 64, Filter1x1Pad0, 16, 16, "
        "4, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 32, 64, Filter1x1Pad0, 32, 32, "
        "1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 16, 64, Filter1x1Pad0, 16, 16, "
        "2, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Pad0, 16, 16, 1, "
        "1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 64, 64, Filter1x1Pad0, 16, 16, "
        "1, 2, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 64, 64, Filter1x1Pad0, 32, 32, "
        "1, 1, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 128, 64, Filter1x1Pad0, 16, 16, "
        "1, 4, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 128, 64, Filter1x1Pad0, 32, 32, "
        "1, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 16, 256, 64, Filter1x1Pad0, 16, 16, "
        "1, 4, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 32, 256, 64, Filter1x1Pad0, 32, 32, "
        "1, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Stride1Pad0, 16, "
        "16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 32, 64, Filter1x1Stride1Pad0, "
        "32, 32, 2, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 4, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 32, 64, Filter1x1Stride1Pad0, "
        "32, 32, 2, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 4, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 32, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 2, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Stride1Pad0, 16, "
        "16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 64, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 2, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 64, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 1, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 128, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 4, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 128, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 16, 256, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 4, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 32, 256, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Default, 16, 16, 1, 1, "
        "8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Default, 16, 16, 1, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Default, 16, 16, 1, 1, 8, "
        "8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Default, 16, 16, 1, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 32, 64, Default, 32, 32, 2, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 16, 64, Default, 16, 16, 4, 1, "
        "8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 32, 64, Default, 32, 32, 2, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 16, 64, Default, 16, 16, 4, 1, "
        "8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 32, 64, Default, 32, 32, 1, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 16, 64, Default, 16, 16, 2, 1, "
        "8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Default, 16, 16, 1, 1, "
        "8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Default, 16, 16, 1, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Default, 16, 16, 1, 1, 8, "
        "8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Default, 16, 16, 1, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 64, 64, Default, 16, 16, 1, 2, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 64, 64, Default, 32, 32, 1, 1, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 128, 64, Default, 16, 16, 1, 4, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 128, 64, Default, 32, 32, 1, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 16, 256, 64, Default, 16, 16, 1, 4, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 32, 256, 64, Default, 32, 32, 1, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Pad0, 16, 16, 1, "
        "1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 32, 64, Filter1x1Pad0, 32, 32, "
        "2, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 16, 64, Filter1x1Pad0, 16, 16, "
        "4, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 32, 64, Filter1x1Pad0, 32, 32, "
        "2, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 16, 64, Filter1x1Pad0, 16, 16, "
        "4, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 32, 64, Filter1x1Pad0, 32, 32, "
        "1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 16, 64, Filter1x1Pad0, 16, 16, "
        "2, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Pad0, 16, 16, 1, "
        "1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 64, 64, Filter1x1Pad0, 16, 16, "
        "1, 2, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 64, 64, Filter1x1Pad0, 32, 32, "
        "1, 1, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 128, 64, Filter1x1Pad0, 16, 16, "
        "1, 4, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 128, 64, Filter1x1Pad0, 32, 32, "
        "1, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 16, 256, 64, Filter1x1Pad0, 16, 16, "
        "1, 4, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 32, 256, 64, Filter1x1Pad0, 32, 32, "
        "1, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Stride1Pad0, 16, "
        "16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 32, 64, Filter1x1Stride1Pad0, "
        "32, 32, 2, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 4, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 32, 64, Filter1x1Stride1Pad0, "
        "32, 32, 2, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 4, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 32, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 2, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Stride1Pad0, 16, "
        "16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 64, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 2, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 64, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 1, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 128, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 4, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 128, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 16, 256, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 4, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 32, 256, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>"};
};

struct Fp32PassThrough
{
    using DataType                                                    = float;
    using Epilogue                                                    = PassThrough;
    static constexpr int COUNT                                        = 173;
    static constexpr std::array<std::string_view, COUNT> TYPE_STRINGS = {
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 16, Default, 32, 32, 2, 2, 1, 1, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Default, 32, 32, 2, 1, 4, 4, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 16, Default, 32, 32, 2, 2, 1, "
        "1, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 16, Default, 32, 32, 4, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 16, Default, 32, 32, 2, 4, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 16, Default, 32, 32, 4, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 16, Default, 32, 32, 2, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 16, Default, 32, 32, 2, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 16, Default, 32, 32, 2, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 16, Default, 32, 32, 2, 2, 4, 4, "
        "4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 16, Default, 32, 32, 2, 1, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 16, Default, 32, 32, 1, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 16, Default, 32, 32, 2, 1, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 16, Default, 32, 32, 1, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Default, 32, 32, 2, 1, 4, 4, "
        "4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 16, Default, 32, 32, 1, 2, 4, 4, "
        "4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 16, Filter1x1Pad0, 32, 32, 2, 2, "
        "1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Filter1x1Pad0, 32, 32, 2, 1, "
        "4, 4, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 16, Filter1x1Pad0, 32, 32, 2, "
        "2, 1, 1, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 16, Filter1x1Pad0, 32, 32, 4, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 16, Filter1x1Pad0, 32, 32, 2, "
        "4, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 16, Filter1x1Pad0, 32, 32, 4, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 16, Filter1x1Pad0, 32, 32, 2, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 16, Filter1x1Pad0, 32, 32, 2, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 16, Filter1x1Pad0, 32, 32, 2, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 16, Filter1x1Pad0, 32, 32, 2, 2, "
        "4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 16, Filter1x1Pad0, 32, 32, 2, "
        "1, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 16, Filter1x1Pad0, 32, 32, 1, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 16, Filter1x1Pad0, 32, 32, 2, "
        "1, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 16, Filter1x1Pad0, 32, 32, 1, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Filter1x1Pad0, 32, 32, 2, 1, "
        "4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 16, Filter1x1Pad0, 32, 32, 1, 2, "
        "4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 4, 4, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 16, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 4, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 16, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 16, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 16, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 16, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Default, 16, 16, 2, 2, 1, "
        "2, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Default, 16, 16, 2, 2, 2, "
        "1, 2, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Default, 16, 16, 2, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Pad0, 16, 16, 2, "
        "2, 1, 2, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Pad0, 16, 16, 2, "
        "2, 2, 1, 2, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Pad0, 16, 16, 2, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Stride1Pad0, 16, "
        "16, 2, 2, 1, 2, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Stride1Pad0, 16, "
        "16, 2, 2, 2, 1, 2, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Stride1Pad0, 16, "
        "16, 2, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<64, 64, 64, 16, Default, 32, 32, "
        "2, 2, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<256, 256, 128, 16, Default, 32, "
        "32, 4, 2, 4, 4, 4, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 16, Default, 16, 16, 4, 1, 4, 1, "
        "1, 1, 1, 8>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 16, Default, 16, 16, 4, 1, 4, 1, "
        "1, 1, 1, 16>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 16, Default, 16, 16, 4, 1, 4, 1, "
        "1, 1, 1, 32>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 16, Filter3x3, 16, 16, 4, 1, 4, "
        "1, 1, 1, 1, 8>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 16, Filter3x3, 16, 16, 4, 1, 4, "
        "1, 1, 1, 1, 16>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 16, Filter3x3, 16, 16, 4, 1, 4, "
        "1, 1, 1, 1, 32>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 32, Default, 32, 32, 2, 2, "
        "4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Default, 32, 32, 2, 2, "
        "4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Default, 32, 32, 2, 2, "
        "4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v5>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Default, 32, 32, 2, 2, "
        "4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 32, Filter1x1Pad0, 32, 32, "
        "2, 2, 4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Filter1x1Pad0, 32, 32, "
        "2, 2, 4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Filter1x1Pad0, 32, 32, "
        "2, 2, 4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v5>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Filter1x1Pad0, 32, 32, "
        "2, 2, 4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 32, Filter1x1Stride1Pad0, "
        "32, 32, 2, 2, 4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v4>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Filter1x1Stride1Pad0, "
        "32, 32, 2, 2, 4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Filter1x1Stride1Pad0, "
        "32, 32, 2, 2, 4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v5>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Filter1x1Stride1Pad0, "
        "32, 32, 2, 2, 4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Default, 16, 16, 1, 1, "
        "4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Default, 16, 16, 1, 1, "
        "4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Default, 16, 16, 1, 1, 4, "
        "4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Default, 16, 16, 1, 1, "
        "4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 32, 64, Default, 32, 32, 2, 1, "
        "4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 16, 64, Default, 16, 16, 4, 1, "
        "4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 32, 64, Default, 32, 32, 1, 1, "
        "4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 16, 64, Default, 16, 16, 2, 1, "
        "4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Default, 16, 16, 1, 1, "
        "4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Default, 16, 16, 1, 1, "
        "4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Default, 16, 16, 1, 1, 4, "
        "4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Default, 16, 16, 1, 1, "
        "4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 64, 64, Default, 16, 16, 1, 2, "
        "4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 64, 64, Default, 32, 32, 1, 1, "
        "4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 128, 64, Default, 16, 16, 1, 4, "
        "4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 128, 64, Default, 32, 32, 1, 2, "
        "4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Pad0, 16, 16, "
        "1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Pad0, 16, 16, 1, "
        "1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 32, 64, Filter1x1Pad0, 32, 32, "
        "2, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 16, 64, Filter1x1Pad0, 16, 16, "
        "4, 1, 4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 32, 64, Filter1x1Pad0, 32, 32, "
        "1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 16, 64, Filter1x1Pad0, 16, 16, "
        "2, 1, 4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Pad0, 16, 16, "
        "1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Pad0, 16, 16, 1, "
        "1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 64, 64, Filter1x1Pad0, 16, 16, "
        "1, 2, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 64, 64, Filter1x1Pad0, 32, 32, "
        "1, 1, 4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 128, 64, Filter1x1Pad0, 16, 16, "
        "1, 4, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 128, 64, Filter1x1Pad0, 32, 32, "
        "1, 2, 4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Stride1Pad0, 16, "
        "16, 1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 32, 64, Filter1x1Stride1Pad0, "
        "32, 32, 2, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 4, 1, 4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 32, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 2, 1, 4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Stride1Pad0, 16, "
        "16, 1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 64, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 2, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 64, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 1, 4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 128, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 4, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 128, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 2, 4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Default, 16, 16, 1, 1, "
        "4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Default, 16, 16, 1, 1, "
        "4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Default, 16, 16, 1, 1, 4, "
        "4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Default, 16, 16, 1, 1, "
        "4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 32, 64, Default, 32, 32, 2, 1, "
        "4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 16, 64, Default, 16, 16, 4, 1, "
        "4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 32, 64, Default, 32, 32, 1, 1, "
        "4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 16, 64, Default, 16, 16, 2, 1, "
        "4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Default, 16, 16, 1, 1, "
        "4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Default, 16, 16, 1, 1, "
        "4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Default, 16, 16, 1, 1, 4, "
        "4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Default, 16, 16, 1, 1, "
        "4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 64, 64, Default, 16, 16, 1, 2, "
        "4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 64, 64, Default, 32, 32, 1, 1, "
        "4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 128, 64, Default, 16, 16, 1, 4, "
        "4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 128, 64, Default, 32, 32, 1, 2, "
        "4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Pad0, 16, 16, "
        "1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Pad0, 16, 16, 1, "
        "1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 32, 64, Filter1x1Pad0, 32, 32, "
        "2, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 16, 64, Filter1x1Pad0, 16, 16, "
        "4, 1, 4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 32, 64, Filter1x1Pad0, 32, 32, "
        "1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 16, 64, Filter1x1Pad0, 16, 16, "
        "2, 1, 4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Pad0, 16, 16, "
        "1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Pad0, 16, 16, 1, "
        "1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 64, 64, Filter1x1Pad0, 16, 16, "
        "1, 2, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 64, 64, Filter1x1Pad0, 32, 32, "
        "1, 1, 4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 128, 64, Filter1x1Pad0, 16, 16, "
        "1, 4, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 128, 64, Filter1x1Pad0, 32, 32, "
        "1, 2, 4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Stride1Pad0, 16, "
        "16, 1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 32, 64, Filter1x1Stride1Pad0, "
        "32, 32, 2, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 4, 1, 4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 32, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 2, 1, 4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Stride1Pad0, 16, "
        "16, 1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 64, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 2, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 64, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 1, 4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 128, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 4, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 128, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 2, 4, 4, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>"};
};

struct Bf16PassThrough
{
    using DataType                                                    = ck::bhalf_t;
    using Epilogue                                                    = PassThrough;
    static constexpr int COUNT                                        = 237;
    static constexpr std::array<std::string_view, COUNT> TYPE_STRINGS = {
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Default, 32, 32, 2, 2, 1, 1, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Default, 32, 32, 2, 1, 8, 8, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Default, 32, 32, 2, 2, 1, "
        "1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Default, 32, 32, 4, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Default, 32, 32, 2, 4, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Default, 32, 32, 4, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Default, 32, 32, 2, 2, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Default, 32, 32, 2, 1, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Default, 32, 32, 1, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Default, 32, 32, 2, 1, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Default, 32, 32, 1, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Default, 32, 32, 2, 1, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Default, 32, 32, 1, 2, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Pad0, 32, 32, 2, 2, "
        "1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Pad0, 32, 32, 2, 1, "
        "8, 8, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 1, 1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Filter1x1Pad0, 32, 32, 4, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Filter1x1Pad0, 32, 32, 2, "
        "4, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Filter1x1Pad0, 32, 32, 4, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Pad0, 32, 32, 2, 2, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Filter1x1Pad0, 32, 32, 2, "
        "1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Filter1x1Pad0, 32, 32, 1, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Filter1x1Pad0, 32, 32, 2, "
        "1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Filter1x1Pad0, 32, 32, 1, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Pad0, 32, 32, 2, 1, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Filter1x1Pad0, 32, 32, 1, 2, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 4, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Default, 16, 16, 2, 2, 1, "
        "2, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Default, 16, 16, 2, 2, 2, "
        "1, 2, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Default, 16, 16, 2, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Default, 16, 16, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Pad0, 16, 16, 2, "
        "2, 1, 2, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Pad0, 16, 16, 2, "
        "2, 2, 1, 2, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Pad0, 16, 16, 2, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Pad0, 16, 16, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Stride1Pad0, 16, "
        "16, 2, 2, 1, 2, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Stride1Pad0, 16, "
        "16, 2, 2, 2, 1, 2, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Stride1Pad0, 16, "
        "16, 2, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Stride1Pad0, 16, "
        "16, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<64, 64, 64, 32, Default, 32, 32, "
        "2, 2, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<256, 256, 128, 32, Default, 32, "
        "32, 4, 2, 2, 2, 2, 1, 1>",
        "DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<256, 256, 128, 32, Default, 32, "
        "32, 4, 2, 8, 8, 8, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 16, Default, 16, 16, 4, 1, 4, 1, "
        "1, 1, 1, 8>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 16, Default, 16, 16, 4, 1, 4, 1, "
        "1, 1, 1, 16>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 16, Default, 16, 16, 4, 1, 4, 1, "
        "1, 1, 1, 32>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 16, Filter3x3, 16, 16, 4, 1, 4, "
        "1, 1, 1, 1, 8>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 16, Filter3x3, 16, 16, 4, 1, 4, "
        "1, 1, 1, 1, 16>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 16, Filter3x3, 16, 16, 4, 1, 4, "
        "1, 1, 1, 1, 32>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Default, 32, 32, 4, 4, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Filter1x1Pad0, 32, 32, "
        "4, 4, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Filter1x1Stride1Pad0, "
        "32, 32, 4, 4, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v4>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Default, 16, 16, 8, 8, "
        "8, 8, 8, 1, 2, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Filter1x1Pad0, 16, 16, "
        "8, 8, 8, 8, 8, 1, 2, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Filter1x1Stride1Pad0, "
        "16, 16, 8, 8, 8, 8, 8, 1, 2, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Default, 32, 32, 2, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Filter1x1Pad0, 32, 32, "
        "2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Filter1x1Stride1Pad0, "
        "32, 32, 2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v4>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 224, 256, 64, Default, 16, 16, 7, 8, "
        "8, 8, 8, 1, 2, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 224, 256, 64, Filter1x1Pad0, 16, 16, "
        "7, 8, 8, 8, 8, 1, 2, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 224, 256, 64, Filter1x1Stride1Pad0, "
        "16, 16, 7, 8, 8, 8, 8, 1, 2, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 32, Default, 32, 32, 2, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 32, Filter1x1Pad0, 32, 32, "
        "2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 32, Filter1x1Stride1Pad0, "
        "32, 32, 2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v4>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 224, 64, Default, 16, 16, 8, 7, "
        "8, 8, 8, 2, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 224, 64, Filter1x1Pad0, 16, 16, "
        "8, 7, 8, 8, 8, 2, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 224, 64, Filter1x1Stride1Pad0, "
        "16, 16, 8, 7, 8, 8, 8, 2, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Default, 32, 32, 4, 4, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Filter1x1Pad0, 32, 32, "
        "4, 4, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Filter1x1Stride1Pad0, "
        "32, 32, 4, 4, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Default, 32, 32, 2, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Filter1x1Pad0, 32, 32, "
        "2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Filter1x1Stride1Pad0, "
        "32, 32, 2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Default, 32, 32, 4, 4, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v5>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Filter1x1Pad0, 32, 32, "
        "4, 4, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v5>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Filter1x1Stride1Pad0, "
        "32, 32, 4, 4, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v5>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Default, 32, 32, 2, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v5>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Filter1x1Pad0, 32, 32, "
        "2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v5>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Filter1x1Stride1Pad0, "
        "32, 32, 2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v5>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 256, 32, Default, 32, 32, 2, 4, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 256, 32, Filter1x1Pad0, 32, 32, "
        "2, 4, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 256, 32, Filter1x1Stride1Pad0, "
        "32, 32, 2, 4, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 128, 32, Default, 32, 32, 4, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 128, 32, Filter1x1Pad0, 32, 32, "
        "4, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 128, 32, Filter1x1Stride1Pad0, "
        "32, 32, 4, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Default, 32, 32, 2, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Filter1x1Pad0, 32, 32, "
        "2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, Filter1x1Stride1Pad0, "
        "32, 32, 2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 64, 64, Default, 32, 32, 2, 1, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 64, 64, Filter1x1Pad0, 32, 32, "
        "2, 1, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 64, 64, Filter1x1Stride1Pad0, "
        "32, 32, 2, 1, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 64, 128, 64, Default, 32, 32, 1, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 64, 128, 64, Filter1x1Pad0, 32, 32, "
        "1, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 64, 128, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 64, 64, 64, Default, 32, 32, 1, 1, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 64, 64, 64, Filter1x1Pad0, 32, 32, "
        "1, 1, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 64, 64, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 1, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v3>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Default, 16, 16, 1, 1, "
        "8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Default, 16, 16, 1, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Default, 16, 16, 1, 1, 8, "
        "8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Default, 16, 16, 1, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 32, 64, Default, 32, 32, 2, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 16, 64, Default, 16, 16, 4, 1, "
        "8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 32, 64, Default, 32, 32, 2, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 16, 64, Default, 16, 16, 4, 1, "
        "8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 32, 64, Default, 32, 32, 1, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 16, 64, Default, 16, 16, 2, 1, "
        "8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Default, 16, 16, 1, 1, "
        "8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Default, 16, 16, 1, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Default, 16, 16, 1, 1, 8, "
        "8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Default, 16, 16, 1, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 64, 64, Default, 16, 16, 1, 2, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 64, 64, Default, 32, 32, 1, 1, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 128, 64, Default, 16, 16, 1, 4, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 128, 64, Default, 32, 32, 1, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 16, 256, 64, Default, 16, 16, 1, 4, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 32, 256, 64, Default, 32, 32, 1, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Pad0, 16, 16, 1, "
        "1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 32, 64, Filter1x1Pad0, 32, 32, "
        "2, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 16, 64, Filter1x1Pad0, 16, 16, "
        "4, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 32, 64, Filter1x1Pad0, 32, 32, "
        "2, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 16, 64, Filter1x1Pad0, 16, 16, "
        "4, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 32, 64, Filter1x1Pad0, 32, 32, "
        "1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 16, 64, Filter1x1Pad0, 16, 16, "
        "2, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Pad0, 16, 16, 1, "
        "1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 64, 64, Filter1x1Pad0, 16, 16, "
        "1, 2, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 64, 64, Filter1x1Pad0, 32, 32, "
        "1, 1, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 128, 64, Filter1x1Pad0, 16, 16, "
        "1, 4, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 128, 64, Filter1x1Pad0, 32, 32, "
        "1, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 16, 256, 64, Filter1x1Pad0, 16, 16, "
        "1, 4, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 32, 256, 64, Filter1x1Pad0, 32, 32, "
        "1, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Stride1Pad0, 16, "
        "16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 32, 64, Filter1x1Stride1Pad0, "
        "32, 32, 2, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 4, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 32, 64, Filter1x1Stride1Pad0, "
        "32, 32, 2, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 4, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 32, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 2, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Stride1Pad0, 16, "
        "16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 64, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 2, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 64, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 1, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 128, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 4, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 128, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 16, 256, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 4, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 32, 256, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Default, 16, 16, 1, 1, "
        "8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Default, 16, 16, 1, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Default, 16, 16, 1, 1, 8, "
        "8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Default, 16, 16, 1, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 32, 64, Default, 32, 32, 2, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 16, 64, Default, 16, 16, 4, 1, "
        "8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 32, 64, Default, 32, 32, 2, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 16, 64, Default, 16, 16, 4, 1, "
        "8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 32, 64, Default, 32, 32, 1, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 16, 64, Default, 16, 16, 2, 1, "
        "8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Default, 16, 16, 1, 1, "
        "8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Default, 16, 16, 1, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Default, 16, 16, 1, 1, 8, "
        "8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Default, 16, 16, 1, 1, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 64, 64, Default, 16, 16, 1, 2, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 64, 64, Default, 32, 32, 1, 1, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 128, 64, Default, 16, 16, 1, 4, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 128, 64, Default, 32, 32, 1, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 16, 256, 64, Default, 16, 16, 1, 4, "
        "8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 32, 256, 64, Default, 32, 32, 1, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Pad0, 16, 16, 1, "
        "1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 32, 64, Filter1x1Pad0, 32, 32, "
        "2, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 16, 64, Filter1x1Pad0, 16, 16, "
        "4, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 32, 64, Filter1x1Pad0, 32, 32, "
        "2, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 16, 64, Filter1x1Pad0, 16, 16, "
        "4, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 32, 64, Filter1x1Pad0, 32, 32, "
        "1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 16, 64, Filter1x1Pad0, 16, 16, "
        "2, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Pad0, 16, 16, 1, "
        "1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Pad0, 16, 16, "
        "1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 64, 64, Filter1x1Pad0, 16, 16, "
        "1, 2, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 64, 64, Filter1x1Pad0, 32, 32, "
        "1, 1, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 128, 64, Filter1x1Pad0, 16, 16, "
        "1, 4, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 128, 64, Filter1x1Pad0, 32, 32, "
        "1, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 16, 256, 64, Filter1x1Pad0, 16, 16, "
        "1, 4, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 32, 256, 64, Filter1x1Pad0, 32, 32, "
        "1, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Stride1Pad0, 16, "
        "16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 32, 64, Filter1x1Stride1Pad0, "
        "32, 32, 2, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 4, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 32, 64, Filter1x1Stride1Pad0, "
        "32, 32, 2, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 128, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 4, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 32, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 64, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 2, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Stride1Pad0, 16, "
        "16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 64, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 2, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 64, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 1, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 128, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 4, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 128, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 16, 256, 64, Filter1x1Stride1Pad0, "
        "16, 16, 1, 4, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 32, 256, 64, Filter1x1Stride1Pad0, "
        "32, 32, 1, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, "
        "BlkGemmPipelineVersion: v2>"};
};

struct S8PassThrough
{
    using DataType                                                    = int8_t;
    using Epilogue                                                    = PassThrough;
    static constexpr int COUNT                                        = 48;
    static constexpr std::array<std::string_view, COUNT> TYPE_STRINGS = {
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Default, 32, 32, 2, 2, 1, 1, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Default, 32, 32, 2, 1, 8, 8, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Default, 32, 32, 2, 2, 1, "
        "1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Default, 32, 32, 4, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Default, 32, 32, 2, 4, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Default, 32, 32, 4, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Default, 32, 32, 2, 2, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Default, 32, 32, 2, 1, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Default, 32, 32, 1, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Default, 32, 32, 2, 1, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Default, 32, 32, 1, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Default, 32, 32, 2, 1, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Default, 32, 32, 1, 2, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Pad0, 32, 32, 2, 2, "
        "1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Pad0, 32, 32, 2, 1, "
        "8, 8, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 1, 1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Filter1x1Pad0, 32, 32, 4, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Filter1x1Pad0, 32, 32, 2, "
        "4, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Filter1x1Pad0, 32, 32, 4, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Pad0, 32, 32, 2, 2, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Filter1x1Pad0, 32, 32, 2, "
        "1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Filter1x1Pad0, 32, 32, 1, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Filter1x1Pad0, 32, 32, 2, "
        "1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Filter1x1Pad0, 32, 32, 1, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Pad0, 32, 32, 2, 1, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Filter1x1Pad0, 32, 32, 1, 2, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 4, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>"};
};

struct Fp16Scale
{
    using DataType                                                    = ck::half_t;
    using Epilogue                                                    = Scale;
    static constexpr int COUNT                                        = 48;
    static constexpr std::array<std::string_view, COUNT> TYPE_STRINGS = {
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Default, 32, 32, 2, 2, 1, 1, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Default, 32, 32, 2, 1, 8, 8, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Default, 32, 32, 2, 2, 1, "
        "1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Default, 32, 32, 4, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Default, 32, 32, 2, 4, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Default, 32, 32, 4, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Default, 32, 32, 2, 2, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Default, 32, 32, 2, 1, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Default, 32, 32, 1, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Default, 32, 32, 2, 1, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Default, 32, 32, 1, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Default, 32, 32, 2, 1, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Default, 32, 32, 1, 2, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Pad0, 32, 32, 2, 2, "
        "1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Pad0, 32, 32, 2, 1, "
        "8, 8, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 1, 1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Filter1x1Pad0, 32, 32, 4, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Filter1x1Pad0, 32, 32, 2, "
        "4, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Filter1x1Pad0, 32, 32, 4, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Pad0, 32, 32, 2, 2, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Filter1x1Pad0, 32, 32, 2, "
        "1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Filter1x1Pad0, 32, 32, 1, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Filter1x1Pad0, 32, 32, 2, "
        "1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Filter1x1Pad0, 32, 32, 1, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Pad0, 32, 32, 2, 1, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Filter1x1Pad0, 32, 32, 1, 2, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 4, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>"};
};

struct Fp32Scale
{
    using DataType                                                    = float;
    using Epilogue                                                    = Scale;
    static constexpr int COUNT                                        = 48;
    static constexpr std::array<std::string_view, COUNT> TYPE_STRINGS = {
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 16, Default, 32, 32, 2, 2, 1, 1, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Default, 32, 32, 2, 1, 4, 4, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 16, Default, 32, 32, 2, 2, 1, "
        "1, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 16, Default, 32, 32, 4, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 16, Default, 32, 32, 2, 4, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 16, Default, 32, 32, 4, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 16, Default, 32, 32, 2, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 16, Default, 32, 32, 2, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 16, Default, 32, 32, 2, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 16, Default, 32, 32, 2, 2, 4, 4, "
        "4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 16, Default, 32, 32, 2, 1, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 16, Default, 32, 32, 1, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 16, Default, 32, 32, 2, 1, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 16, Default, 32, 32, 1, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Default, 32, 32, 2, 1, 4, 4, "
        "4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 16, Default, 32, 32, 1, 2, 4, 4, "
        "4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 16, Filter1x1Pad0, 32, 32, 2, 2, "
        "1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Filter1x1Pad0, 32, 32, 2, 1, "
        "4, 4, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 16, Filter1x1Pad0, 32, 32, 2, "
        "2, 1, 1, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 16, Filter1x1Pad0, 32, 32, 4, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 16, Filter1x1Pad0, 32, 32, 2, "
        "4, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 16, Filter1x1Pad0, 32, 32, 4, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 16, Filter1x1Pad0, 32, 32, 2, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 16, Filter1x1Pad0, 32, 32, 2, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 16, Filter1x1Pad0, 32, 32, 2, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 16, Filter1x1Pad0, 32, 32, 2, 2, "
        "4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 16, Filter1x1Pad0, 32, 32, 2, "
        "1, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 16, Filter1x1Pad0, 32, 32, 1, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 16, Filter1x1Pad0, 32, 32, 2, "
        "1, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 16, Filter1x1Pad0, 32, 32, 1, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Filter1x1Pad0, 32, 32, 2, 1, "
        "4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 16, Filter1x1Pad0, 32, 32, 1, 2, "
        "4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 4, 4, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 16, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 4, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 16, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 16, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 16, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 16, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 4, 4, 4, 1, 1, 1>"};
};

struct Bf16Scale
{
    using DataType                                                    = ck::bhalf_t;
    using Epilogue                                                    = Scale;
    static constexpr int COUNT                                        = 48;
    static constexpr std::array<std::string_view, COUNT> TYPE_STRINGS = {
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Default, 32, 32, 2, 2, 1, 1, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Default, 32, 32, 2, 1, 8, 8, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Default, 32, 32, 2, 2, 1, "
        "1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Default, 32, 32, 4, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Default, 32, 32, 2, 4, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Default, 32, 32, 4, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Default, 32, 32, 2, 2, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Default, 32, 32, 2, 1, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Default, 32, 32, 1, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Default, 32, 32, 2, 1, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Default, 32, 32, 1, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Default, 32, 32, 2, 1, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Default, 32, 32, 1, 2, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Pad0, 32, 32, 2, 2, "
        "1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Pad0, 32, 32, 2, 1, "
        "8, 8, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 1, 1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Filter1x1Pad0, 32, 32, 4, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Filter1x1Pad0, 32, 32, 2, "
        "4, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Filter1x1Pad0, 32, 32, 4, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Pad0, 32, 32, 2, 2, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Filter1x1Pad0, 32, 32, 2, "
        "1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Filter1x1Pad0, 32, 32, 1, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Filter1x1Pad0, 32, 32, 2, "
        "1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Filter1x1Pad0, 32, 32, 1, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Pad0, 32, 32, 2, 1, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Filter1x1Pad0, 32, 32, 1, 2, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 4, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>"};
};

struct S8Scale
{
    using DataType                                                    = int8_t;
    using Epilogue                                                    = Scale;
    static constexpr int COUNT                                        = 48;
    static constexpr std::array<std::string_view, COUNT> TYPE_STRINGS = {
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Default, 32, 32, 2, 2, 1, 1, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Default, 32, 32, 2, 1, 8, 8, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Default, 32, 32, 2, 2, 1, "
        "1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Default, 32, 32, 4, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Default, 32, 32, 2, 4, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Default, 32, 32, 4, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Default, 32, 32, 2, 2, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Default, 32, 32, 2, 1, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Default, 32, 32, 1, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Default, 32, 32, 2, 1, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Default, 32, 32, 1, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Default, 32, 32, 2, 1, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Default, 32, 32, 1, 2, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Pad0, 32, 32, 2, 2, "
        "1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Pad0, 32, 32, 2, 1, "
        "8, 8, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 1, 1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Filter1x1Pad0, 32, 32, 4, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Filter1x1Pad0, 32, 32, 2, "
        "4, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Filter1x1Pad0, 32, 32, 4, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Pad0, 32, 32, 2, 2, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Filter1x1Pad0, 32, 32, 2, "
        "1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Filter1x1Pad0, 32, 32, 1, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Filter1x1Pad0, 32, 32, 2, "
        "1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Filter1x1Pad0, 32, 32, 1, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Pad0, 32, 32, 2, 1, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Filter1x1Pad0, 32, 32, 1, 2, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 4, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>"};
};

struct Fp16Bilinear
{
    using DataType                                                    = ck::half_t;
    using Epilogue                                                    = Bilinear;
    static constexpr int COUNT                                        = 48;
    static constexpr std::array<std::string_view, COUNT> TYPE_STRINGS = {
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Default, 32, 32, 2, 2, 1, 1, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Default, 32, 32, 2, 1, 8, 8, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Default, 32, 32, 2, 2, 1, "
        "1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Default, 32, 32, 4, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Default, 32, 32, 2, 4, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Default, 32, 32, 4, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Default, 32, 32, 2, 2, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Default, 32, 32, 2, 1, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Default, 32, 32, 1, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Default, 32, 32, 2, 1, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Default, 32, 32, 1, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Default, 32, 32, 2, 1, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Default, 32, 32, 1, 2, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Pad0, 32, 32, 2, 2, "
        "1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Pad0, 32, 32, 2, 1, "
        "8, 8, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 1, 1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Filter1x1Pad0, 32, 32, 4, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Filter1x1Pad0, 32, 32, 2, "
        "4, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Filter1x1Pad0, 32, 32, 4, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Pad0, 32, 32, 2, 2, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Filter1x1Pad0, 32, 32, 2, "
        "1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Filter1x1Pad0, 32, 32, 1, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Filter1x1Pad0, 32, 32, 2, "
        "1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Filter1x1Pad0, 32, 32, 1, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Pad0, 32, 32, 2, 1, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Filter1x1Pad0, 32, 32, 1, 2, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 4, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>"};
};

struct Fp32Bilinear
{
    using DataType                                                    = float;
    using Epilogue                                                    = Bilinear;
    static constexpr int COUNT                                        = 48;
    static constexpr std::array<std::string_view, COUNT> TYPE_STRINGS = {
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 16, Default, 32, 32, 2, 2, 1, 1, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Default, 32, 32, 2, 1, 4, 4, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 16, Default, 32, 32, 2, 2, 1, "
        "1, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 16, Default, 32, 32, 4, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 16, Default, 32, 32, 2, 4, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 16, Default, 32, 32, 4, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 16, Default, 32, 32, 2, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 16, Default, 32, 32, 2, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 16, Default, 32, 32, 2, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 16, Default, 32, 32, 2, 2, 4, 4, "
        "4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 16, Default, 32, 32, 2, 1, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 16, Default, 32, 32, 1, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 16, Default, 32, 32, 2, 1, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 16, Default, 32, 32, 1, 2, 4, "
        "4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Default, 32, 32, 2, 1, 4, 4, "
        "4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 16, Default, 32, 32, 1, 2, 4, 4, "
        "4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 16, Filter1x1Pad0, 32, 32, 2, 2, "
        "1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Filter1x1Pad0, 32, 32, 2, 1, "
        "4, 4, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 16, Filter1x1Pad0, 32, 32, 2, "
        "2, 1, 1, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 16, Filter1x1Pad0, 32, 32, 4, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 16, Filter1x1Pad0, 32, 32, 2, "
        "4, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 16, Filter1x1Pad0, 32, 32, 4, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 16, Filter1x1Pad0, 32, 32, 2, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 16, Filter1x1Pad0, 32, 32, 2, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 16, Filter1x1Pad0, 32, 32, 2, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 16, Filter1x1Pad0, 32, 32, 2, 2, "
        "4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 16, Filter1x1Pad0, 32, 32, 2, "
        "1, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 16, Filter1x1Pad0, 32, 32, 1, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 16, Filter1x1Pad0, 32, 32, 2, "
        "1, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 16, Filter1x1Pad0, 32, 32, 1, "
        "2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Filter1x1Pad0, 32, 32, 2, 1, "
        "4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 16, Filter1x1Pad0, 32, 32, 1, 2, "
        "4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 4, 4, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 16, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 4, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 16, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 16, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 16, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 4, 4, 4, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 16, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 4, 4, 4, 1, 1, 1>"};
};

struct Bf16Bilinear
{
    using DataType                                                    = ck::bhalf_t;
    using Epilogue                                                    = Bilinear;
    static constexpr int COUNT                                        = 48;
    static constexpr std::array<std::string_view, COUNT> TYPE_STRINGS = {
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Default, 32, 32, 2, 2, 1, 1, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Default, 32, 32, 2, 1, 8, 8, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Default, 32, 32, 2, 2, 1, "
        "1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Default, 32, 32, 4, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Default, 32, 32, 2, 4, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Default, 32, 32, 4, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Default, 32, 32, 2, 2, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Default, 32, 32, 2, 1, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Default, 32, 32, 1, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Default, 32, 32, 2, 1, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Default, 32, 32, 1, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Default, 32, 32, 2, 1, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Default, 32, 32, 1, 2, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Pad0, 32, 32, 2, 2, "
        "1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Pad0, 32, 32, 2, 1, "
        "8, 8, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 1, 1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Filter1x1Pad0, 32, 32, 4, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Filter1x1Pad0, 32, 32, 2, "
        "4, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Filter1x1Pad0, 32, 32, 4, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Pad0, 32, 32, 2, 2, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Filter1x1Pad0, 32, 32, 2, "
        "1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Filter1x1Pad0, 32, 32, 1, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Filter1x1Pad0, 32, 32, 2, "
        "1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Filter1x1Pad0, 32, 32, 1, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Pad0, 32, 32, 2, 1, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Filter1x1Pad0, 32, 32, 1, 2, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 4, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>"};
};

struct S8Bilinear
{
    using DataType                                                    = int8_t;
    using Epilogue                                                    = Bilinear;
    static constexpr int COUNT                                        = 48;
    static constexpr std::array<std::string_view, COUNT> TYPE_STRINGS = {
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Default, 32, 32, 2, 2, 1, 1, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Default, 32, 32, 2, 1, 8, 8, "
        "1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Default, 32, 32, 2, 2, 1, "
        "1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Default, 32, 32, 4, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Default, 32, 32, 2, 4, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Default, 32, 32, 4, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Default, 32, 32, 2, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Default, 32, 32, 2, 2, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Default, 32, 32, 2, 1, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Default, 32, 32, 1, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Default, 32, 32, 2, 1, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Default, 32, 32, 1, 2, 8, "
        "8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Default, 32, 32, 2, 1, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Default, 32, 32, 1, 2, 8, 8, "
        "8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Pad0, 32, 32, 2, 2, "
        "1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Pad0, 32, 32, 2, 1, "
        "8, 8, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 1, 1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Filter1x1Pad0, 32, 32, 4, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Filter1x1Pad0, 32, 32, 2, "
        "4, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Filter1x1Pad0, 32, 32, 4, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Filter1x1Pad0, 32, 32, 2, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Pad0, 32, 32, 2, 2, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Filter1x1Pad0, 32, 32, 2, "
        "1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Filter1x1Pad0, 32, 32, 1, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Filter1x1Pad0, 32, 32, 2, "
        "1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Filter1x1Pad0, 32, 32, 1, "
        "2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Pad0, 32, 32, 2, 1, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Filter1x1Pad0, 32, 32, 1, 2, "
        "8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 1, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 1, 1, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 256, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 4, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 4, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 64, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 32, 128, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Stride1Pad0, 32, "
        "32, 2, 1, 8, 8, 8, 1, 1, 1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 32, 64, 32, Filter1x1Stride1Pad0, 32, "
        "32, 1, 2, 8, 8, 8, 1, 1, 1>"};
};

using MyTypes = ::testing::Types<Fp16PassThrough,
                                 Fp32PassThrough,
                                 Bf16PassThrough,
                                 S8PassThrough,
                                 Fp16Scale,
                                 Fp32Scale,
                                 Bf16Scale,
                                 S8Scale,
                                 Fp16Bilinear,
                                 Fp32Bilinear,
                                 Bf16Bilinear,
                                 S8Bilinear>;

TYPED_TEST_SUITE_P(CkFactoryTest);

TYPED_TEST_P(CkFactoryTest, CreatesGrpConvFwdInstances)
{
    using DataType = typename TypeParam::DataType;
    using Epilogue = typename TypeParam::Epilogue;
    auto conv_ptrs = DeviceOpGFwdPtrs<DataType, Epilogue>::GetInstances();
    EXPECT_EQ(conv_ptrs.size(), TypeParam::COUNT);
    for(int i = 0; i < conv_ptrs.size(); ++i)
    {
        const auto& conv_ptr = conv_ptrs[i];
        EXPECT_TRUE(conv_ptr != nullptr);
        if(!conv_ptr)
            continue;
        EXPECT_EQ(conv_ptr->GetTypeString(),
                  (i < TypeParam::COUNT) ? TypeParam::TYPE_STRINGS[i] : std::string_view());
    }
}

REGISTER_TYPED_TEST_SUITE_P(CkFactoryTest, CreatesGrpConvFwdInstances);
INSTANTIATE_TYPED_TEST_SUITE_P(Full, CkFactoryTest, MyTypes);

} // namespace
} // namespace miopen::unit_test
#endif
