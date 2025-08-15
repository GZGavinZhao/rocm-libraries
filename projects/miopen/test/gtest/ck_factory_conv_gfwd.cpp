// Unit tests using gtest for the CK grouped convolution forward factory.
//
// ## Purpose:
// Checks that the CK factory for grouped convolution forward operations (with different
// epilogue types: PassThrough, Scale, Bilinear) creates the expected numbers of instances for the
// tested data types (fp16, fp32, bf16, s8).
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
// - Data types:
//   - fp16: ck::half_t
//   - fp32: float
//   - bf16: ck::bhalf_t
//   - s8: int8_t
// - Epilogues: PassThrough, Scale, Bilinear
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

template <typename T>
class CkFactoryTest : public testing::Test
{
};

struct Fp16PassThrough
{
    using DataType             = ck::half_t;
    using Epilogue             = PassThrough;
    static constexpr int COUNT = 228;
};

struct Fp32PassThrough
{
    using DataType             = float;
    using Epilogue             = PassThrough;
    static constexpr int COUNT = 173;
};

struct Bf16PassThrough
{
    using DataType             = ck::bhalf_t;
    using Epilogue             = PassThrough;
    static constexpr int COUNT = 237;
};

struct S8PassThrough
{
    using DataType             = int8_t;
    using Epilogue             = PassThrough;
    static constexpr int COUNT = 48;
};

struct Fp16Scale
{
    using DataType             = ck::half_t;
    using Epilogue             = Scale;
    static constexpr int COUNT = 48;
};

struct Fp32Scale
{
    using DataType             = float;
    using Epilogue             = Scale;
    static constexpr int COUNT = 48;
};

struct Bf16Scale
{
    using DataType             = ck::bhalf_t;
    using Epilogue             = Scale;
    static constexpr int COUNT = 48;
};

struct S8Scale
{
    using DataType             = int8_t;
    using Epilogue             = Scale;
    static constexpr int COUNT = 48;
};

struct Fp16Bilinear
{
    using DataType             = ck::half_t;
    using Epilogue             = Bilinear;
    static constexpr int COUNT = 48;
};

struct Fp32Bilinear
{
    using DataType             = float;
    using Epilogue             = Bilinear;
    static constexpr int COUNT = 48;
};

struct Bf16Bilinear
{
    using DataType             = ck::bhalf_t;
    using Epilogue             = Bilinear;
    static constexpr int COUNT = 48;
};

struct S8Bilinear
{
    using DataType             = int8_t;
    using Epilogue             = Bilinear;
    static constexpr int COUNT = 48;
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
    ASSERT_EQ(conv_ptrs.size(), TypeParam::COUNT);
}

REGISTER_TYPED_TEST_SUITE_P(CkFactoryTest, CreatesGrpConvFwdInstances);
INSTANTIATE_TYPED_TEST_SUITE_P(Full, CkFactoryTest, MyTypes);

} // namespace
} // namespace miopen::unit_test
#endif
