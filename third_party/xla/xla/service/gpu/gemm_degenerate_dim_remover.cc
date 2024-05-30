/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/gemm_degenerate_dim_remover.h"

#include <cstdint>
#include <numeric>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

// Construct a new layout by adding removing the minor-most dimension to the
// input layout. For example, {3, 2, 1, 0} is extended to {2, 1, 0}.
// We expect that the input layout is normalized by LayoutNormalizer, so that
// the input layout has a descending ordering.
absl::StatusOr<Layout> GetLayoutWithNewMinorMostDimension(
    const Layout& layout) {
  if (!LayoutUtil::IsMonotonicWithDim0Major(layout)) {
    return absl::InvalidArgumentError("Layout is not normalized.");
  }
  return LayoutUtil::MakeDescendingLayout(layout.minor_to_major_size() - 1);
}

absl::StatusOr<HloInstruction*>
NormalizeDotOperandToBatchMajorAndContractingMinor(
    HloInstruction* dot_operand, absl::Span<const int64_t> batch_dimensions,
    absl::Span<const int64_t> contracting_dimensions) {
  std::vector<int64_t> transpose_dimensions(batch_dimensions.begin(),
                                            batch_dimensions.end());
  for (int64_t i = 0; i < dot_operand->shape().rank(); ++i) {
    if (!(absl::c_linear_search(batch_dimensions, i) ||
          absl::c_linear_search(contracting_dimensions, i))) {
      transpose_dimensions.push_back(i);
    }
  }
  transpose_dimensions.insert(transpose_dimensions.end(),
                              contracting_dimensions.begin(),
                              contracting_dimensions.end());
  if (absl::c_is_sorted(transpose_dimensions)) {
    return dot_operand;
  }
  return MakeTransposeHlo(dot_operand, transpose_dimensions);
}

// The GemvRewriter creates a bitcast which adds a degenerate dimension as the
// last dimension.
bool IsDegenerateDimensionAddedByGemvRewriter(const HloInstruction* bitcast) {
  absl::Span<const int64_t> bitcast_dims = bitcast->shape().dimensions();
  absl::Span<const int64_t> bitcast_operand_dims =
      bitcast->operand(0)->shape().dimensions();
  if (bitcast_dims.size() != bitcast_operand_dims.size() + 1) {
    return false;
  }
  for (int64_t i = 0; i < bitcast_operand_dims.size(); ++i) {
    if (bitcast_dims[i] != bitcast_operand_dims[i]) {
      return false;
    }
  }
  return true;
}

class GemmDegenerateDimRemoverVisitor : public DfsHloRewriteVisitor {
 public:
  explicit GemmDegenerateDimRemoverVisitor(
      GemmDegenerateDimRemover* dim_remover)
      : dim_remover_(dim_remover) {}

  absl::Status HandleDot(HloInstruction* instr) override {
    HloDotInstruction* dot = Cast<HloDotInstruction>(instr);
    HloInstruction* lhs = dot->mutable_operand(0);
    HloInstruction* rhs = dot->mutable_operand(1);

    HloInstruction* new_lhs = nullptr;
    HloInstruction* new_rhs = nullptr;

    // The degenerate dimension is the last dimension of the LHS or RHS.
    if (lhs->shape().dimensions().back() == 1) {
      if (lhs->opcode() != HloOpcode::kBitcast) {
        return absl::OkStatus();
      }
      if (!IsDegenerateDimensionAddedByGemvRewriter(lhs)) {
        return absl::OkStatus();
      }
      new_lhs = lhs->mutable_operand(0);
      new_rhs = rhs;
    } else if (rhs->shape().dimensions().back() == 1) {
      if (rhs->opcode() != HloOpcode::kBitcast) {
        return absl::OkStatus();
      }
      if (!IsDegenerateDimensionAddedByGemvRewriter(rhs)) {
        return absl::OkStatus();
      }
      new_lhs = lhs;
      new_rhs = rhs->mutable_operand(0);
    } else {
      return absl::OkStatus();
    }

    changed_ = true;

    std::vector<int64_t> new_out_dimensions;
    new_out_dimensions.reserve(dot->shape().dimensions().size() - 1);
    for (int64_t dim_size : dot->shape().dimensions()) {
      if (dim_size == 1) {
        continue;
      }
      new_out_dimensions.push_back(dim_size);
    }

    // GemvRewriter should only add one degenerate dimension.
    if (new_out_dimensions.size() != dot->shape().dimensions().size() - 1) {
      return absl::InternalError(
          "More than one degenerate dimension in the output shape.");
    }

    Shape new_out_shape(
        dot->shape().element_type(), new_out_dimensions,
        absl::InlinedVector<bool, 4>(new_out_dimensions.size(), false),
        /*tuple_shapes=*/{});
    TF_ASSIGN_OR_RETURN(
        *new_out_shape.mutable_layout(),
        GetLayoutWithNewMinorMostDimension(dot->shape().layout()));

    HloComputation* computation = dot->parent();
    HloInstruction* new_dot =
        computation->AddInstruction(HloInstruction::CreateDot(
            new_out_shape, new_lhs, new_rhs, dot->dot_dimension_numbers(),
            dot->precision_config()));

    if (dot->user_count() != 1) {
      return absl::InternalError("Dot should have exactly one user.");
    }
    HloInstruction* bitcast = dot->users()[0];
    if (bitcast->opcode() != HloOpcode::kBitcast) {
      return absl::InternalError("Dot user should be a bitcast.");
    }
    TF_RETURN_IF_ERROR(computation->ReplaceInstruction(bitcast, new_dot));
    return StrengthReduceDot(Cast<HloDotInstruction>(new_dot));
  }

  bool changed() const { return changed_; }

 private:
  // If the lhs or rhs have only batch and contracting dimensions, a dot can be
  // rewritten as reduce(mul(broadcast(transpose(x)),broadcast(transpose(y))))
  absl::Status StrengthReduceDot(HloDotInstruction* dot) {
    HloInstruction* lhs = dot->mutable_operand(0);
    HloInstruction* rhs = dot->mutable_operand(1);
    const auto& dnums = dot->dot_dimension_numbers();

    TF_ASSIGN_OR_RETURN(HloInstruction * new_lhs,
                        NormalizeDotOperandToBatchMajorAndContractingMinor(
                            lhs, dnums.lhs_batch_dimensions(),
                            dnums.lhs_contracting_dimensions()));
    if (!ShapeUtil::SameElementType(dot->shape(), new_lhs->shape())) {
      new_lhs = MakeConvertToHlo(new_lhs, dot->shape().element_type());
    }

    TF_ASSIGN_OR_RETURN(HloInstruction * new_rhs,
                        NormalizeDotOperandToBatchMajorAndContractingMinor(
                            rhs, dnums.rhs_batch_dimensions(),
                            dnums.rhs_contracting_dimensions()));
    if (!ShapeUtil::SameElementType(dot->shape(), new_rhs->shape())) {
      new_rhs = MakeConvertToHlo(new_rhs, dot->shape().element_type());
    }

    int64_t lhs_outer_dims =
        lhs->shape().rank() - (dnums.lhs_batch_dimensions_size() +
                               dnums.lhs_contracting_dimensions_size());
    int64_t rhs_outer_dims =
        rhs->shape().rank() - (dnums.rhs_batch_dimensions_size() +
                               dnums.rhs_contracting_dimensions_size());
    CHECK(lhs_outer_dims == 0 || rhs_outer_dims == 0);
    if (rhs_outer_dims > 0) {
      std::vector<int64_t> lhs_broadcast_dims(
          dnums.lhs_batch_dimensions_size());
      absl::c_iota(lhs_broadcast_dims, 0);
      lhs_broadcast_dims.resize(lhs->shape().rank());
      std::iota(lhs_broadcast_dims.begin() + dnums.lhs_batch_dimensions_size(),
                lhs_broadcast_dims.end(),
                dnums.lhs_batch_dimensions_size() + rhs_outer_dims);
      new_lhs = dot->AddInstruction(HloInstruction::CreateBroadcast(
          new_rhs->shape(), new_lhs, lhs_broadcast_dims));
    } else if (lhs_outer_dims > 0) {
      std::vector<int64_t> rhs_broadcast_dims(
          dnums.rhs_batch_dimensions_size());
      absl::c_iota(rhs_broadcast_dims, 0);
      rhs_broadcast_dims.resize(rhs->shape().rank());
      std::iota(rhs_broadcast_dims.begin() + dnums.rhs_batch_dimensions_size(),
                rhs_broadcast_dims.end(),
                dnums.rhs_batch_dimensions_size() + lhs_outer_dims);
      new_rhs = dot->AddInstruction(HloInstruction::CreateBroadcast(
          new_lhs->shape(), new_rhs, rhs_broadcast_dims));
    }

    TF_ASSIGN_OR_RETURN(HloInstruction * new_dot,
                        MakeBinaryHlo(HloOpcode::kMultiply, new_lhs, new_rhs));
    std::vector<int64_t> reduce_dims(dnums.lhs_contracting_dimensions_size());
    PrimitiveType dot_type =
        ShapeUtil::ElementIsFloating(dot->shape())
            ? (dot->shape().element_type() == F64 ? F64 : F32)
            : dot->shape().element_type();
    new_dot = AsType(new_dot, dot_type);
    const int64_t outer_dims = std::max(rhs_outer_dims, lhs_outer_dims);
    absl::c_iota(reduce_dims, outer_dims + dnums.lhs_batch_dimensions_size());
    new_dot = AddReduce(new_dot, reduce_dims, dot_type);
    new_dot = AsType(new_dot, dot->shape().element_type());
    return ReplaceInstruction(dot, new_dot);
  }

  // Converts to primitive type if the input hlo is not that type, otherwise
  // returns the original hlo.
  HloInstruction* AsType(HloInstruction* hlo,
                         const PrimitiveType element_type) {
    if (hlo->shape().element_type() == element_type) {
      return hlo;
    }
    Shape changed_shape =
        ShapeUtil::ChangeElementType(hlo->shape(), element_type);
    dim_remover_->UpdateLayout(&changed_shape);
    return hlo->parent()->AddInstruction(
        HloInstruction::CreateConvert(changed_shape, hlo));
  }

  HloComputation* GetOrCreateScalarAddComputation(HloModule* module,
                                                  PrimitiveType type) {
    HloComputation*& scalar_add_computation = scalar_add_computations_[type];
    if (scalar_add_computation) {
      return scalar_add_computation;
    }

    HloComputation::Builder b("scalar_add_computation");
    Shape shape = ShapeUtil::MakeShape(type, {});
    dim_remover_->UpdateLayout(&shape);
    auto scalar_lhs = b.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "scalar_lhs"));
    auto scalar_rhs = b.AddInstruction(
        HloInstruction::CreateParameter(1, shape, "scalar_rhs"));
    auto scalar_op = b.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kAdd, scalar_lhs, scalar_rhs));
    scalar_add_computation = module->AddEmbeddedComputation(b.Build(scalar_op));
    return scalar_add_computation;
  }

  HloInstruction* AddReduce(HloInstruction* hlo, absl::Span<const int64_t> dims,
                            PrimitiveType type) {
    HloInstruction* zero = hlo->parent()->AddInstruction(
        dim_remover_->CreateConstantWithLayoutUpdated(
            LiteralUtil::Zero(hlo->shape().element_type()).Clone()));
    HloComputation* AddReduce_computation =
        GetOrCreateScalarAddComputation(hlo->parent()->parent(), type);
    Shape shape = ShapeUtil::DeleteDimensions(dims, hlo->shape());
    dim_remover_->UpdateLayout(&shape);
    return hlo->parent()->AddInstruction(HloInstruction::CreateReduce(
        shape, hlo, zero, dims, AddReduce_computation));
  }

  GemmDegenerateDimRemover* dim_remover_;
  // Cached computation for adding two scalars of a given type.
  absl::flat_hash_map<PrimitiveType, HloComputation*> scalar_add_computations_;
  bool changed_ = false;
};

}  // namespace

absl::StatusOr<bool> GemmDegenerateDimRemover::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  GemmDegenerateDimRemoverVisitor visitor(this);
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  }
  std::cerr << module->ToString() << "\n";
  return visitor.changed();
}

}  // namespace gpu
}  // namespace xla
