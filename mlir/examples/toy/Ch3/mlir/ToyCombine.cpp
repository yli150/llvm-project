//===- ToyCombine.cpp - Toy High Level Optimizer --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a set of simple combiners for optimizing operations in
// the Toy dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/Operation.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Diagnostics.h"
#include "toy/Dialect.h"

using namespace mlir;
using namespace toy;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "ToyCombine.inc"
} // namespace

/// This is an example of a c++ rewrite pattern for the TransposeOp. It
/// optimizes the following scenario: transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

    // Input defined by another transpose? If not, no match.
    if (!transposeInputOp)
      return failure();

    // Otherwise, we have a redundant transpose. Use the rewriter.
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}



struct SimplifyRedundantReshape : public mlir::OpRewritePattern<ReshapeOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantReshape(mlir::MLIRContext *context)
      : OpRewritePattern<ReshapeOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(ReshapeOp origOp,
                  mlir::PatternRewriter &rewriter) const override {
    // auto inputType = llvm::dyn_cast<RankedTensorType>(origOp.getOperand().getType());
    auto resultType = llvm::dyn_cast<RankedTensorType>(origOp.getResult().getType());
    auto outShape = resultType.getShape();
    // Only erase the op with output shape first dim is one. 
    if (outShape[0] == 1) {
          rewriter.replaceOp(origOp, {origOp.getOperand()});
    }

    return success();
  }
};


struct FuseReshape : public mlir::OpRewritePattern<ReshapeOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  FuseReshape(mlir::MLIRContext *context)
      : OpRewritePattern<ReshapeOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(ReshapeOp origOp,
                  mlir::PatternRewriter &rewriter) const override {
    
    auto prevOp = origOp.getOperand().getDefiningOp<ReshapeOp>();
    if (prevOp == nullptr) {
        return mlir::failure();
    }
    // Fusing AffineReshape with any of the above mentioned ops might result in another AffineReshape or not,
    // depending on the resulting input and output shapes.
    // If the Reshape that replaces the two ops ends up being a valid AffineReshape, then it will be converted by
    // Reshape's canonicalizer.
    // Pay attention to ReshapeOp::build interface generated under "/build/tools/mlir/examples/toy/Ch2/include/toy/Ops.cpp.inc"
    // void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value input) {
    rewriter.replaceOpWithNewOp<ReshapeOp>(origOp, origOp.getResult().getType(), prevOp->getOperand(0));
    // rewriter.replaceOp(origOp, {origOp.getOperand()});

    return success();
  }
};


/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<FuseReshape>(context);
  // results.add<SimplifyRedundantReshape>(context);

}

SmallVector<int64_t> getI64SubArrayToy(ArrayAttr arrayAttr,
                                          unsigned dropFront,
                                          unsigned dropBack) {
  assert(arrayAttr.size() > dropFront + dropBack && "Out of bounds");
  auto range = arrayAttr.getAsRange<IntegerAttr>();
  SmallVector<int64_t> res;
  res.reserve(arrayAttr.size() - dropFront - dropBack);
  for (auto it = range.begin() + dropFront, eit = range.end() - dropBack;
       it != eit; ++it)
    res.push_back((*it).getValue().getSExtValue());
  return res;
}

struct FusePermute : public mlir::OpRewritePattern<PermuteOp> {

  FusePermute(mlir::MLIRContext *context)
      : OpRewritePattern<PermuteOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(PermuteOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.
    auto prevPermute = op.getOperand().getDefiningOp<PermuteOp>();
    // Input defined by another transpose? If not, no match.
    if (!prevPermute)
      return failure();

    // Check output matches previous permute input type 
    auto outType = llvm::dyn_cast<RankedTensorType>(op.getResult().getType());
    auto prePermuteInputType = llvm::dyn_cast<RankedTensorType>(prevPermute.getOperand().getType());
    if (outType != prePermuteInputType){
      return failure();
    }

    // Check Perm 
    llvm::SmallVector<int64_t> expectedShape = applyPermutation(prePermuteInputType.getShape(), 
            getI64SubArrayToy(prevPermute.getPerm(), 0, 0));

    llvm::SmallVector<int64_t> expectedOutShape = applyPermutation(expectedShape, 
            getI64SubArrayToy(op.getPerm(), 0, 0));

    if (llvm::equal(outType.getShape(), expectedOutShape)) {
      mlir::emitWarning(op.getLoc()) << "expectedOutShape " << expectedOutShape;
      rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getResult().getType(), prevPermute->getOperand(0));
      return success();
    }

    return success();
  }
};


void PermuteOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<FusePermute>(context);
}
