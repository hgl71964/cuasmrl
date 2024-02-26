#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <memory>
#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

using namespace mlir;

class TritonGPUOptimizeThreadLocalityPass
    : public TritonGPUOptimizeThreadLocalityBase<
          TritonGPUOptimizeThreadLocalityPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    DenseSet<triton::ReduceOp> reduceOps;
    mod.walk([&](triton::ReduceOp reduce) -> void {
      auto srcType = reduce.getOperands()[0].getType().cast<RankedTensorType>();
      auto rank = srcType.getShape().size();
      auto srcEncoding = srcType.getEncoding();
      auto reductionOp = getReductionOp(reduce);
      if (!reductionOp ||
          !isa<arith::AddFOp, arith::MaximumFOp, arith::MinimumFOp,
               arith::MulFOp>(reductionOp.value()))
        return;
      // TODO: relax this restriction
      if (!(srcEncoding.isa<triton::gpu::BlockedEncodingAttr>() && rank > 1))
        return;
      for (auto operand : reduce->getOperands()) {
        auto def = operand.getDefiningOp();
        if (!isa<triton::LoadOp>(def))
          return;
      }
      auto elemsPerThread =
          triton::gpu::getElemsPerThread(srcType)[reduce.getAxis()];
      // Not worth applying this optimization if there is only one element per
      // thread on the reduction axis
      if (elemsPerThread == 1)
        return;
      if (!reduce->hasOneUse())
        return;
      Operation *user = *(reduce->getUsers().begin());
      if (!user->hasOneUse())
        return;
      OpOperand &yieldOpOperand = *(user->getUses().begin());
      auto yieldOp = dyn_cast<scf::YieldOp>(yieldOpOperand.getOwner());
      if (!yieldOp)
        return;
      auto operandNumber = yieldOpOperand.getOperandNumber();
      Block *block = reduce->getBlock();
      Operation *parentOp = block->getParentOp();
      auto forOp = dyn_cast<scf::ForOp>(parentOp);
      if (!forOp)
        return;
      auto argNum = yieldOpOperand.getOperandNumber();
      auto oldAccum = forOp.getInitArgs()[argNum];
      auto cstOp = dyn_cast<arith::ConstantOp>(oldAccum.getDefiningOp());
      if (!cstOp)
        return;
      reduceOps.insert(reduce);
    });

    for (auto reduce : reduceOps) {
      OpBuilder builder(reduce);
      auto srcType = reduce.getOperands()[0].getType().cast<RankedTensorType>();
      auto srcShape = srcType.getShape();
      auto srcEncoding = srcType.getEncoding();
      assert(srcEncoding.isa<triton::gpu::BlockedEncodingAttr>() &&
             "Thread locality optimization only supports blocked encoding");
      auto blocked = srcEncoding.dyn_cast<triton::gpu::BlockedEncodingAttr>();
      auto elemsPerThread =
          triton::gpu::getElemsPerThread(srcType)[reduce.getAxis()];
      auto rank = srcShape.size();
      // create new layouts
      auto blocked3d = getThreadLocalityOptimizedEncoding(reduce);
      auto viewOpTensorShape = getThreadLocalityOptimizedShape(reduce);
      auto viewOpTensorType = RankedTensorType::get(
          viewOpTensorShape, srcType.getElementType(), blocked3d);
      auto slice2d = triton::gpu::SliceEncodingAttr::get(mod.getContext(), rank,
                                                         blocked3d);
      // Get forOp
      assert(reduce->hasOneUse());
      OpOperand &use = *(reduce->getUses().begin());
      auto operandNumber = use.getOperandNumber();
      auto oldUpdate = use.getOwner();
      assert(oldUpdate->getNumOperands() == 2);
      auto accumOperandNumber = (operandNumber == 0) ? 1 : 0;
      auto accumOperand = oldUpdate->getOperand(accumOperandNumber);
      assert(accumOperand.isa<BlockArgument>());
      auto blockArg = accumOperand.dyn_cast<BlockArgument>();
      auto blockArgNum = blockArg.getArgNumber();
      auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
      // get oldAccum
      auto oldAccum =
          forOp.getInitArgs()[blockArgNum - forOp.getNumInductionVars()];
      // get old loop user
      Value loopResult =
          forOp.getResult(blockArgNum - forOp.getNumInductionVars());
      assert(loopResult.hasOneUse());
      OpOperand &loopUse = *(loopResult.getUses().begin());
      Operation *loopUser = loopUse.getOwner();
      // get old loop yield
      auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      // create newAccum initialization
      auto newAccum =
          createAccum(builder, reduce, oldAccum, viewOpTensorShape, slice2d);
      // create new loop by copying the old for op signature and appending
      // newAccum to the block arguments
      auto newLoop = replaceForOpWithNewSignature(
          builder, forOp, ValueRange{newAccum->getResult(0)});
      // create thread local reduction (also adds viewOps)
      auto newReduce = createReduce(builder, reduce, viewOpTensorType);

      // create new accum update
      auto newUpdate = createUpdate(builder, newLoop, newReduce, oldUpdate);
      // create new yield
      auto newYield = createYield(builder, newLoop, oldYield,
                                  newUpdate->getResult(0), blockArgNum);
      // create post loop reduction on the original reduce axis
      auto newReduce2 = createPostLoopReduce(builder, newLoop, reduce);
      // add convert_layout to get back to original layout, the result layout
      // should now match the layout of the old accumulator (%cst)
      Type destType = loopResult.getType();
      auto cvtLayout = createConvertLayout(builder, destType, newReduce2);
      // incorporate the original accumulator value into the final result
      auto finalOp = incorporateOriginalAccumulatorValue(builder, oldUpdate,
                                                         cvtLayout, oldAccum);
      // Replace the old loop user with the final result
      loopUser->setOperand(loopUse.getOperandNumber(), finalOp->getResult(0));

      // cleanup
      oldYield.erase();
      forOp.erase();
    }
  };

private:
  std::optional<Operation *> getReductionOp(triton::ReduceOp reduce) const {
    auto numRegions = reduce->getNumRegions();
    if (numRegions != 1)
      return std::nullopt;
    Region &region = reduce->getRegion(0);
    auto numBlocks = region.getBlocks().size();
    if (numBlocks != 1)
      return std::nullopt;
    Block &block = region.front();
    auto blockWithoutTerminator = block.without_terminator();
    auto blockSizeWithoutTerminator = std::distance(
        blockWithoutTerminator.begin(), blockWithoutTerminator.end());
    if (blockSizeWithoutTerminator != 1)
      return std::nullopt;
    Operation *op = &block.front();
    return std::optional<Operation *>(op);
  }
  Operation *incorporateOriginalAccumulatorValue(OpBuilder &builder,
                                                 Operation *oldUpdate,
                                                 Operation *cvtLayout,
                                                 Value oldAccum) const {
    builder.setInsertionPointAfter(cvtLayout);
    IRMapping mapping;
    mapping.map(oldUpdate->getOperand(0), oldAccum);
    mapping.map(oldUpdate->getOperand(1), cvtLayout->getResult(0));
    auto finalOp = cloneWithInferType(builder, &(*oldUpdate), mapping);
    return finalOp;
  }
  Operation *createConvertLayout(OpBuilder &builder, Type destType,
                                 Operation *newReduce) const {
    builder.setInsertionPointAfter(newReduce);
    auto newCvt = builder.create<triton::gpu::ConvertLayoutOp>(
        newReduce->getLoc(), destType, newReduce->getResult(0));
    return newCvt;
  }

  Operation *createPostLoopReduce(OpBuilder &builder, scf::ForOp &loop,
                                  triton::ReduceOp &reduce) const {
    auto resultIndex =
        loop.getBody()->getNumArguments() - 1 - loop.getNumInductionVars();
    auto newLoopResult = loop.getResult(resultIndex);
    builder.setInsertionPointAfter(loop);
    IRMapping mapping;
    mapping.map(*(reduce.getOperands().begin()), newLoopResult);
    auto newReduce2 = cloneWithInferType(builder, &(*reduce), mapping);
    return newReduce2;
  }

  Operation *createYield(OpBuilder &builder, scf::ForOp &loop,
                         scf::YieldOp &oldYield, Value newUpdate,
                         int oldAccumBlockArgNum) const {
    builder.setInsertionPoint(oldYield);
    SmallVector<Value> yieldValues = llvm::to_vector(oldYield.getOperands());
    yieldValues[oldAccumBlockArgNum - 1] =
        loop.getBody()->getArgument(oldAccumBlockArgNum);
    yieldValues.push_back(newUpdate);
    auto newYield =
        builder.create<scf::YieldOp>(oldYield.getLoc(), yieldValues);
    return newYield;
  }

  Operation *createUpdate(OpBuilder &builder, scf::ForOp &loop,
                          Operation *newReduce, Operation *oldUpdate) const {
    auto blockArgNum = loop.getBody()->getNumArguments() - 1;
    auto newArg = loop.getBody()->getArgument(blockArgNum);
    builder.setInsertionPointAfter(newReduce);
    IRMapping mapping;
    mapping.map(oldUpdate->getOperand(0), newArg);
    mapping.map(oldUpdate->getOperand(1), newReduce->getResult(0));
    auto newUpdate = cloneWithInferType(builder, oldUpdate, mapping);
    return newUpdate;
  }

  Operation *createReduce(OpBuilder &builder, triton::ReduceOp reduce,
                          Type viewOpTensorType) const {
    auto srcType = reduce.getOperands()[0].getType().cast<RankedTensorType>();
    auto rank = srcType.getShape().size();
    builder.setInsertionPointAfter(reduce);
    IRMapping mapping;
    for (auto operand : reduce.getOperands()) {
      auto viewOp = builder.create<triton::ViewOp>(reduce.getLoc(),
                                                   viewOpTensorType, operand);
      mapping.map(operand, viewOp);
    }

    auto newReduce = cloneWithInferType(builder, &(*reduce), mapping);
    newReduce->setAttr("axis", builder.getI32IntegerAttr(rank));
    auto typeInfer = dyn_cast<InferTypeOpInterface>(newReduce);
    if (typeInfer) {
      SmallVector<Type, 1> newTypes;
      auto success = typeInfer.inferReturnTypes(
          newReduce->getContext(), newReduce->getLoc(),
          newReduce->getOperands(), newReduce->getAttrDictionary(),
          newReduce->getPropertiesStorage(), newReduce->getRegions(), newTypes);
      if (succeeded(success)) {
        for (size_t i = 0; i < newTypes.size(); i++)
          newReduce->getResult(i).setType(newTypes[i]);
      }
    }
    return newReduce;
  }

  Operation *createAccum(OpBuilder &builder, triton::ReduceOp reduce,
                         Value &oldAccum, SmallVector<int64_t> &shape,
                         Attribute &slice2d) const {
    // Drop the last dimension (thread locality dimension)
    SmallVector<int64_t> accumShape(shape.begin(), shape.end() - 1);
    auto elemType =
        oldAccum.getType().cast<RankedTensorType>().getElementType();
    // Create tensor type for the new accumulator
    auto accumType = RankedTensorType::get(accumShape, elemType, slice2d);
    // Create new accumulator
    builder.setInsertionPointAfter(oldAccum.getDefiningOp());
    auto reductionOp = getReductionOp(reduce);
    assert(reductionOp && "Processing a reduce that is not supported!");
    auto neutralVal = mlir::arith::getNeutralElement(reductionOp.value());
    assert(neutralVal && "Could not find neutral value for reduction op!");
    auto denseAttr = DenseElementsAttr::get(accumType, neutralVal.value());
    auto newAccum = builder.create<arith::ConstantOp>(oldAccum.getLoc(),
                                                      accumType, denseAttr);
    return newAccum;
  }

  SmallVector<int64_t>
  getThreadLocalityOptimizedShape(triton::ReduceOp reduce) const {
    auto srcType = reduce.getOperands()[0].getType().cast<RankedTensorType>();
    auto srcShape = srcType.getShape();
    auto rank = srcShape.size();
    auto elemsPerThread =
        triton::gpu::getElemsPerThread(srcType)[reduce.getAxis()];
    auto viewOpTensorShape = insertValue(srcShape, rank, 1);
    viewOpTensorShape[reduce.getAxis()] /= elemsPerThread;
    viewOpTensorShape[rank] = elemsPerThread;
    return viewOpTensorShape;
  }

  Attribute getThreadLocalityOptimizedEncoding(triton::ReduceOp reduce) const {
    auto srcType = reduce.getOperands()[0].getType().cast<RankedTensorType>();
    auto rank = srcType.getShape().size();
    auto srcEncoding = srcType.getEncoding();
    auto blocked = srcEncoding.dyn_cast<triton::gpu::BlockedEncodingAttr>();
    auto sizePerThread3d =
        insertValue(blocked.getSizePerThread(), rank,
                    blocked.getSizePerThread()[reduce.getAxis()]);
    sizePerThread3d[reduce.getAxis()] = 1;
    auto threadsPerWarp3d = insertValue(blocked.getThreadsPerWarp(), rank, 1);
    auto warsPerCTA3d = insertValue(blocked.getWarpsPerCTA(), rank, 1);
    auto order3d = insertValue(blocked.getOrder(), 0, rank);
    auto ctasPerCGA3d =
        insertValue(blocked.getCTALayout().getCTAsPerCGA(), rank, 1);
    auto ctasSplitNum3d =
        insertValue(blocked.getCTALayout().getCTASplitNum(), rank, 1);
    auto ctaOrder3d =
        insertValue(blocked.getCTALayout().getCTAOrder(), rank, rank);
    auto ctaLayout3d = triton::gpu::CTALayoutAttr::get(
        reduce.getContext(), ctasPerCGA3d, ctasSplitNum3d, ctaOrder3d);
    auto blocked3d = triton::gpu::BlockedEncodingAttr::get(
        reduce.getContext(), sizePerThread3d, threadsPerWarp3d, warsPerCTA3d,
        order3d, ctaLayout3d);
    return blocked3d;
  }

  template <typename T>
  SmallVector<T> insertValue(ArrayRef<T> vec, unsigned index, int value) const {
    SmallVector<T> res(vec.begin(), vec.end());
    res.insert(res.begin() + index, static_cast<T>(value));
    return res;
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUOptimizeThreadLocalityPass() {
  return std::make_unique<TritonGPUOptimizeThreadLocalityPass>();
}
