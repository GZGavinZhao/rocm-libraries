/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/layernorm.hpp>
#include <miopen/layernorm/solvers.hpp>
#include <miopen/layernorm/invoke_params.hpp>
#if MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/kernels/ck_header_only/layernorm/normalization_fwd.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include <miopen/solver/ck_utility_common.hpp>
#endif

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_LAYERNORM4DCKFORWARD_CONV_CK_LN)

namespace miopen {
namespace solver {
namespace layernorm {
#if MIOPEN_USE_COMPOSABLEKERNEL

using F16  = ck::half_t;
using F32  = float;
using F64  = double;
using BF16 = ushort;

template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename YDataType,
          typename SaveMeanInvStdDataType>
using DeviceOp = ck::tensor_operation::device::DeviceNormalizationFwd<
    XDataType,
    GammaDataType,
    BetaDataType,
    YDataType,
    SaveMeanInvStdDataType,
    ck::tensor_operation::element_wise::PassThrough,
    4,
    3>;
template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename YDataType,
          typename SaveMeanInvStdDataType>
using DeviceOpLnFwdPtrs = kernels::ck_header_only::layernorm::DeviceOperationInstanceFactory<
    DeviceOp<XDataType, GammaDataType, BetaDataType, YDataType, SaveMeanInvStdDataType>>;

namespace {
struct CKArgs
{
    CKArgs(const miopen::layernorm::ProblemDescription& problem)
    {
        auto length = problem.GetXDesc().GetLengths();

        N = length[0];
        H = length[1];
        W = length[2];
        C = length[3];

        N_stride = H * W * C;
        H_stride = W * C;
        W_stride = C;
        C_stride = 1;

        xyLengths    = {N, H, W, C};
        xyStrides    = {N_stride, H_stride, W_stride, C_stride};
        gammaStrides = {0, H_stride, W_stride, C_stride};
        betaStrides  = {0, H_stride, W_stride, C_stride};
        meanStrides  = {1};
        rstdStrides  = {1};
        epsilon      = problem.GetEpsilon();
    }

    CKArgs(const CKArgs&) = default;
    CKArgs(CKArgs&&)      = default;
    CKArgs& operator=(const CKArgs&) = default;

    template <typename LNPtr, typename LNParams>
    auto MakeArgPtr(const LNPtr& ln_ptr, const LNParams& data_context) const
    {
        return ln_ptr->MakeArgumentPointer(xyLengths,
                                           xyStrides,
                                           gammaStrides,
                                           betaStrides,
                                           xyStrides,
                                           meanStrides,
                                           rstdStrides,
                                           {1, 2, 3},
                                           epsilon,
                                           data_context.x,
                                           data_context.weight,
                                           data_context.bias,
                                           data_context.y,
                                           data_context.mean,
                                           data_context.rstd,
                                           ck::tensor_operation::element_wise::PassThrough{});
    }

    template <typename LNPtr>
    bool IsSupportedBy(const LNPtr& ln_ptr) const
    {
        auto arg_ptr = MakeArgPtr(ln_ptr, miopen::layernorm::InvokeParams{});
        return ln_ptr->IsSupportedArgument(arg_ptr.get());
    }

    int32_t N;
    int32_t C;
    int32_t H;
    int32_t W;
    int32_t N_stride;
    int32_t C_stride;
    int32_t H_stride;
    int32_t W_stride;
    std::vector<int32_t> xyLengths;
    std::vector<int32_t> xyStrides;
    std::vector<int32_t> gammaStrides;
    std::vector<int32_t> betaStrides;
    std::vector<int32_t> meanStrides;
    std::vector<int32_t> rstdStrides;
    float epsilon;
};
} // namespace

template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename YDataType,
          typename SaveMeanInvStdDataType>
void PerformanceConfigLayernorm4DCKForward::Init(
    const miopen::layernorm::ProblemDescription& problem)
{
    const auto& args       = CKArgs{problem};
    const auto ln_fwd_ptrs = DeviceOpLnFwdPtrs<XDataType,
                                               GammaDataType,
                                               BetaDataType,
                                               YDataType,
                                               SaveMeanInvStdDataType>::GetInstances();
    if(ln_fwd_ptrs.empty())
    {
        MIOPEN_THROW(miopenStatusInternalError, "Ln4DCKFwd ln_fwd_ptrs empty");
    }

    for(const auto& it : ln_fwd_ptrs)
    {
        auto argument_ptr = it->MakeArgumentPointer(args.xyLengths,
                                                    args.xyStrides,
                                                    args.gammaStrides,
                                                    args.betaStrides,
                                                    args.xyStrides,
                                                    args.meanStrides,
                                                    args.rstdStrides,
                                                    {1, 2, 3},
                                                    args.epsilon,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    PassThrough{});
        if(it->IsSupportedArgument(argument_ptr.get()))
        {
            valid_kernels.push_back(it->GetTypeString());
        }
    }

    if(valid_kernels.empty())
    {
        MIOPEN_THROW(miopenStatusInternalError, "Ln4DCKFwd valid_kernels empty");
    }

    index     = 0;
    kernel_id = valid_kernels[0];
}

template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename YDataType,
          typename SaveMeanInvStdDataType>
bool PerformanceConfigLayernorm4DCKForward::CheckIsSupportCkArgs(
    const miopen::layernorm::ProblemDescription& problem) const
{
    return IsCKArgsSupported<DeviceOpLnFwdPtrs<XDataType,
                                               GammaDataType,
                                               BetaDataType,
                                               YDataType,
                                               SaveMeanInvStdDataType>,
                             CKArgs>(problem, kernel_id);
}

template <typename DeviceOpType>
bool CheckCKApplicability(const miopen::layernorm::ProblemDescription& problem)
{
    const auto ln_args = CKArgs{problem};
    const auto ln_ptrs = DeviceOpType::GetInstances();

    return std::any_of(ln_ptrs.begin(), ln_ptrs.end(), [&ln_args](auto& ln_ptrs) {
        return ln_args.IsSupportedBy(ln_ptrs);
    });
}

template <typename LnPtrsType>
typename LnPtrsType::iterator FindLnPtr(LnPtrsType& ln_ptrs,
                                        const miopen::layernorm::ProblemDescription& problem)
{
    const auto ln_args = CKArgs{problem};
    return std::find_if(ln_ptrs.begin(), ln_ptrs.end(), [&ln_args](auto& ln_ptrs) {
        return ln_args.IsSupportedBy(ln_ptrs);
    });
}
#endif

void PerformanceConfigLayernorm4DCKForward::HeuristicInit(
    const miopen::layernorm::ProblemDescription& problem)
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = problem;
#else
    switch(problem.GetXDesc().GetType())
    {
    case miopenHalf: Init<F16, F16, F16, F16, F16>(problem); break;
    case miopenFloat: Init<F32, F32, F32, F32, F32>(problem); break;
    case miopenBFloat16:
    case miopenDouble:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt8:
    case miopenInt32:
    case miopenInt64:
    default: MIOPEN_THROW("Unsupported datatype");
    }
#endif
}

bool PerformanceConfigLayernorm4DCKForward::SetNextValue(
    const miopen::layernorm::ProblemDescription& problem)
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = problem;
    return false;
#else
    if(valid_kernels.empty())
    {
        HeuristicInit(problem);
        if(valid_kernels.empty())
        {
            MIOPEN_THROW(miopenStatusInternalError, "Ln4DCKFwd valid_kernels empty");
        }
        return true;
    }
    if(index + 1 < valid_kernels.size())
    {
        ++index;
        kernel_id = valid_kernels[index];
        return true;
    }
    return false;
#endif
}

bool PerformanceConfigLayernorm4DCKForward::IsValidValue() const
{
    return index >= 0 && index < valid_kernels.size();
}

bool PerformanceConfigLayernorm4DCKForward::IsValid(
    const ExecutionContext&, const miopen::layernorm::ProblemDescription& problem) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = problem;
    return false;
#else
    switch(problem.GetXDesc().GetType())
    {
    case miopenHalf: return CheckIsSupportCkArgs<F16, F16, F16, F16, F16>(problem);
    case miopenFloat: return CheckIsSupportCkArgs<F32, F32, F32, F32, F32>(problem);
    case miopenBFloat16:
    case miopenDouble:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt8:
    case miopenInt32:
    case miopenInt64:
    default: MIOPEN_THROW("Unsupported datatype");
    }
    return false;
#endif
}

bool PerformanceConfigLayernorm4DCKForward::operator==(
    const PerformanceConfigLayernorm4DCKForward& other) const
{
    return kernel_id == other.kernel_id;
}

PerformanceConfigLayernorm4DCKForward Layernorm4DCKForward::GetDefaultPerformanceConfig(
    const ExecutionContext&, const miopen::layernorm::ProblemDescription& problem) const
{
    PerformanceConfigLayernorm4DCKForward config;
    config.HeuristicInit(problem);
    MIOPEN_LOG_I(config.ToString());
    return config;
}

bool Layernorm4DCKForward::IsValidPerformanceConfig(
    const ExecutionContext& context,
    const miopen::layernorm::ProblemDescription& problem,
    const PerformanceConfigLayernorm4DCKForward& config) const
{
    return config.IsValid(context, problem);
}

PerformanceConfigLayernorm4DCKForward
Layernorm4DCKForward::Search(const ExecutionContext& context,
                             const miopen::layernorm::ProblemDescription& problem,
                             const AnyInvokeParams& invoke_context) const
{
    return GenericSearch(*this, context, problem, invoke_context);
}

bool IsRank4Dim1(const miopen::layernorm::ProblemDescription& problem)
{
    return (problem.GetXDesc().GetLengths().size() == 4) && (problem.GetNormalizedDim() == 1);
}

bool Layernorm4DCKForward::IsApplicable(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::layernorm::ProblemDescription& problem) const
{
#if MIOPEN_USE_COMPOSABLEKERNEL
    if(env::disabled(MIOPEN_DEBUG_LAYERNORM4DCKFORWARD_CONV_CK_LN))
        return false;
    if(!problem.IsSameType())
        return false;
    if(!problem.IsSameLength())
        return false;
    if(!problem.IsAllPacked())
        return false;
    if(!IsRank4Dim1(problem))
        return false;
    if(!problem.IsLargeSize())
        return false;
    if(!ck_utility::is_ck_whitelist(context.GetStream()))
        return false;

    switch(problem.GetXDesc().GetType())
    {
    case miopenHalf:
        return CheckCKApplicability<DeviceOpLnFwdPtrs<F16, F16, F16, F16, F16>>(problem);
    case miopenFloat:
        return CheckCKApplicability<DeviceOpLnFwdPtrs<F32, F32, F32, F32, F32>>(problem);
    case miopenBFloat16:
    case miopenDouble:
    case miopenInt64:
    case miopenInt32:
    case miopenInt8:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz: return false;
    }
#endif
    return false;
}

ConvSolution Layernorm4DCKForward::GetSolution(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::layernorm::ProblemDescription& problem,
    [[maybe_unused]] const PerformanceConfigLayernorm4DCKForward& config) const
{
#if MIOPEN_USE_COMPOSABLEKERNEL
    switch(problem.GetXDesc().GetType())
    {
    case miopenHalf:
        return InitAnyInvokerFactory<DeviceOpLnFwdPtrs<F16, F16, F16, F16, F16>,
                                     CKArgs,
                                     miopen::layernorm::InvokeParams>(problem, config.kernel_id);
    case miopenFloat:
        return InitAnyInvokerFactory<DeviceOpLnFwdPtrs<F32, F32, F32, F32, F32>,
                                     CKArgs,
                                     miopen::layernorm::InvokeParams>(problem, config.kernel_id);
    case miopenDouble:
    case miopenBFloat16:
    case miopenInt8:
    case miopenInt32:
    case miopenInt64:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    default:
        MIOPEN_THROW(miopenStatusInternalError,
                     "Layernorm4DCKForward operation not implemented for this data type");
    }
#endif
    return {};
}

} // namespace layernorm
} // namespace solver
} // namespace miopen
