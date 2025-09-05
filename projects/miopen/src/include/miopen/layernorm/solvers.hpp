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
#pragma once

#include "miopen/execution_context.hpp"
#include "miopen/generic_search.hpp"
#include "miopen/invoke_params.hpp"
#include "miopen/performance_config.hpp"
#include <miopen/layernorm/problem_description.hpp>
#include <miopen/solver.hpp>

namespace miopen {

namespace solver {

namespace layernorm {

using NormalizationSolver =
    NonTunableSolverBase<ExecutionContext, miopen::layernorm::ProblemDescription>;

template <class PerformanceConfig>
using NormalizationTunableSolver =
    TunableSolverMixin<ExecutionContext, miopen::layernorm::ProblemDescription, PerformanceConfig>;

struct PerformanceConfigLayernorm : PerfConfigBase<PerformanceConfigLayernorm>
{
    int local_size;
    bool initialized = false;
    PerformanceConfigLayernorm(int _local_size) : local_size(_local_size) {}
    PerformanceConfigLayernorm() : PerformanceConfigLayernorm(static_cast<int>(1)) {}
    PerformanceConfigLayernorm(bool) : PerformanceConfigLayernorm(static_cast<int>(1)) {}
    void HeuristicInit(const miopen::layernorm::ProblemDescription& problem);
    bool SetNextValue(const miopen::layernorm::ProblemDescription& problem);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext& context,
                 const miopen::layernorm::ProblemDescription& problem) const;

    template <typename Self, typename F>
    static void Visit(Self&& s, F f)
    {
        f(s.local_size, "local_size");
    }
    bool operator==(const PerformanceConfigLayernorm& other) const;

public:
    static constexpr auto default_local_size = 256;
    static constexpr auto max_local_size = 256;
};

struct LayernormBase : NormalizationTunableSolver<PerformanceConfigLayernorm>
{
    bool IsApplicable(const ExecutionContext& context,
                      const miopen::layernorm::ProblemDescription& problem) const override;
    bool IsDynamic() const override { return true; }
    PerformanceConfigLayernorm GetDefaultPerformanceConfig(
        const ExecutionContext& context,
        const miopen::layernorm::ProblemDescription& problem) const override;
    bool IsValidPerformanceConfig(const ExecutionContext& context,
                                  const miopen::layernorm::ProblemDescription& problem,
                                  const PerformanceConfigLayernorm& config) const override;
};

struct LayernormForward final : LayernormBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<LayernormForward>(); }
    PerformanceConfigLayernorm Search(const ExecutionContext& context,
                                      const miopen::layernorm::ProblemDescription& problem,
                                      const AnyInvokeParams& invoke_context) const override
    {

        return GenericSearch(*this, context, problem, invoke_context);
    }
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::layernorm::ProblemDescription& problem,
                             const PerformanceConfigLayernorm& config) const override;
};

struct PerformanceConfigLayernorm2DCKForward
    : PerfConfigBaseCK<PerformanceConfigLayernorm2DCKForward>
{
    int index;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    PerformanceConfigLayernorm2DCKForward(int _index, std::string _kernel_id)
        : index(_index), kernel_id(_kernel_id)
    {
    }
    PerformanceConfigLayernorm2DCKForward() : PerformanceConfigLayernorm2DCKForward(0, "") {}
    PerformanceConfigLayernorm2DCKForward(bool) : PerformanceConfigLayernorm2DCKForward(0, "") {}
    void HeuristicInit(const miopen::layernorm::ProblemDescription& problem);
    bool SetNextValue(const miopen::layernorm::ProblemDescription& problem);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext& context,
                 const miopen::layernorm::ProblemDescription& problem) const;

    template <typename Self, typename F>
    static void Visit(Self&& s, F f)
    {
        f(s.kernel_id, "kernel_id");
    }
    bool operator==(const PerformanceConfigLayernorm2DCKForward& other) const;

private:
    template <typename XDataType,
              typename GammaDataType,
              typename BetaDataType,
              typename YDataType,
              typename SaveMeanInvStdDataType>
    void Init(const miopen::layernorm::ProblemDescription& problem);
    template <typename XDataType,
              typename GammaDataType,
              typename BetaDataType,
              typename YDataType,
              typename SaveMeanInvStdDataType>
    bool CheckIsSupportCkArgs(const miopen::layernorm::ProblemDescription& problem) const;
};

struct Layernorm2DCKForward final
    : NormalizationTunableSolver<PerformanceConfigLayernorm2DCKForward>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<Layernorm2DCKForward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::layernorm::ProblemDescription& problem) const override;
    bool IsDynamic() const override { return true; }
    PerformanceConfigLayernorm2DCKForward GetDefaultPerformanceConfig(
        const ExecutionContext& context,
        const miopen::layernorm::ProblemDescription& problem) const override;
    bool
    IsValidPerformanceConfig(const ExecutionContext& context,
                             const miopen::layernorm::ProblemDescription& problem,
                             const PerformanceConfigLayernorm2DCKForward& config) const override;
    PerformanceConfigLayernorm2DCKForward
    Search(const ExecutionContext& context,
           const miopen::layernorm::ProblemDescription& problem,
           const AnyInvokeParams& invoke_context) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::layernorm::ProblemDescription& problem,
                             const PerformanceConfigLayernorm2DCKForward& config) const override;
};

struct PerformanceConfigLayernorm4DCKForward
    : PerfConfigBaseCK<PerformanceConfigLayernorm4DCKForward>
{
    int index;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    PerformanceConfigLayernorm4DCKForward(int _index, std::string _kernel_id)
        : index(_index), kernel_id(_kernel_id)
    {
    }
    PerformanceConfigLayernorm4DCKForward() : PerformanceConfigLayernorm4DCKForward(0, "") {}
    PerformanceConfigLayernorm4DCKForward(bool) : PerformanceConfigLayernorm4DCKForward(0, "") {}
    void HeuristicInit(const miopen::layernorm::ProblemDescription& problem);
    bool SetNextValue(const miopen::layernorm::ProblemDescription& problem);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext& context,
                 const miopen::layernorm::ProblemDescription& problem) const;

    template <typename Self, typename F>
    static void Visit(Self&& s, F f)
    {
        f(s.kernel_id, "kernel_id");
    }
    bool operator==(const PerformanceConfigLayernorm4DCKForward& other) const;

private:
    template <typename XDataType,
              typename GammaDataType,
              typename BetaDataType,
              typename YDataType,
              typename SaveMeanInvStdDataType>
    void Init(const miopen::layernorm::ProblemDescription& problem);
    template <typename XDataType,
              typename GammaDataType,
              typename BetaDataType,
              typename YDataType,
              typename SaveMeanInvStdDataType>
    bool CheckIsSupportCkArgs(const miopen::layernorm::ProblemDescription& problem) const;
};

struct Layernorm4DCKForward final
    : NormalizationTunableSolver<PerformanceConfigLayernorm4DCKForward>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<Layernorm4DCKForward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::layernorm::ProblemDescription& problem) const override;
    bool IsDynamic() const override { return true; }
    PerformanceConfigLayernorm4DCKForward GetDefaultPerformanceConfig(
        const ExecutionContext& context,
        const miopen::layernorm::ProblemDescription& problem) const override;
    bool
    IsValidPerformanceConfig(const ExecutionContext& context,
                             const miopen::layernorm::ProblemDescription& problem,
                             const PerformanceConfigLayernorm4DCKForward& config) const override;
    PerformanceConfigLayernorm4DCKForward
    Search(const ExecutionContext& context,
           const miopen::layernorm::ProblemDescription& problem,
           const AnyInvokeParams& invoke_context) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::layernorm::ProblemDescription& problem,
                             const PerformanceConfigLayernorm4DCKForward& config) const override;
};

struct LayernormBackward final : LayernormBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<LayernormBackward>(); }
    PerformanceConfigLayernorm Search(const ExecutionContext& context,
                                      const miopen::layernorm::ProblemDescription& problem,
                                      const AnyInvokeParams& invoke_context) const override
    {
        return GenericSearch(*this, context, problem, invoke_context);
    }
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::layernorm::ProblemDescription& problem,
                             const PerformanceConfigLayernorm& config) const override;
    std::size_t
    GetWorkspaceSize(const ExecutionContext& context,
                     const miopen::layernorm::ProblemDescription& problem) const override;
    bool MayNeedWorkspace() const override { return true; }
};

struct AddLayernormForward final : NormalizationSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<AddLayernormForward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::layernorm::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::layernorm::ProblemDescription& problem) const override;
};

struct T5LayernormForward final : NormalizationSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<T5LayernormForward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::layernorm::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::layernorm::ProblemDescription& problem) const override;
};

struct T5LayernormBackward final : NormalizationSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<T5LayernormBackward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::layernorm::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::layernorm::ProblemDescription& problem) const override;
    std::size_t
    GetWorkspaceSize(const ExecutionContext& context,
                     const miopen::layernorm::ProblemDescription& problem) const override;
    bool MayNeedWorkspace() const override { return true; }
};

} // namespace layernorm

} // namespace solver

} // namespace miopen
