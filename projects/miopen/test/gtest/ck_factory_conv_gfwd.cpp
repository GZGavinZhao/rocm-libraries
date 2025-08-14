// Unit tests using gtest for the CK grouped convolution forward factory.
//
// ## Purpose:
// Checks that the CK factory for grouped convolution forward operations (with different
// epilogue types: PassThrough, Scale, Bilinear) does NOT create any instances for the tested
// data types (float16_t, float32_t).
//
// ## How:
// Each test calls GetInstances() on a factory type for a specific combination of data type and
// epilogue and asserts that the returned vector of instances is empty.
//
// ## Why:
// This verifies that the instance factory is producing consistent output. If we change the
// available kernels we should update this test.
//
// ## Tested Types:
// - Data types: c:float16_t, ck::float32_t
// - Epilogues: PassThrough, Scale, Bilinear
// *note: these are the only types that compiled and linked correctly.*
#include "miopen/config.h"
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <gtest/gtest.h>

#include "miopen/solver/ck_utility_common.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bilinear.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scale.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"

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

TEST(CkFactoryTest, CreatesGrpConvFwdPassThroughF16Instances)
{
    auto conv_ptrs = DeviceOpGFwdPtrs<ck::float16_t, PassThrough>::GetInstances();
    ASSERT_EQ(conv_ptrs.size(), 0);
}

TEST(CkFactoryTest, CreatesGrpConvFwdPassThroughF32Instances)
{
    auto conv_ptrs = DeviceOpGFwdPtrs<ck::float32_t, PassThrough>::GetInstances();
    ASSERT_EQ(conv_ptrs.size(), 0);
}

TEST(CkFactoryTest, CreatesGrpConvFwdScaleF16Instances)
{
    auto conv_ptrs = DeviceOpGFwdPtrs<ck::float16_t, Scale>::GetInstances();
    ASSERT_EQ(conv_ptrs.size(), 0);
}

TEST(CkFactoryTest, CreatesGrpConvFwdScaleF32Instances)
{
    auto conv_ptrs = DeviceOpGFwdPtrs<ck::float32_t, Scale>::GetInstances();
    ASSERT_EQ(conv_ptrs.size(), 0);
}

TEST(CkFactoryTest, CreatesGrpConvFwdBilinearF16Instances)
{
    auto conv_ptrs = DeviceOpGFwdPtrs<ck::float16_t, Bilinear>::GetInstances();
    ASSERT_EQ(conv_ptrs.size(), 0);
}

TEST(CkFactoryTest, CreatesGrpConvFwdBilinearF32Instances)
{
    auto conv_ptrs = DeviceOpGFwdPtrs<ck::float32_t, Bilinear>::GetInstances();
    ASSERT_EQ(conv_ptrs.size(), 0);
}

} // namespace
} // namespace miopen::unit_test
#endif
