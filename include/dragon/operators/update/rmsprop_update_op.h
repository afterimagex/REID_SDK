// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_UPDATE_RMSPROP_UPDATE_OP_H_
#define DRAGON_OPERATORS_UPDATE_RMSPROP_UPDATE_OP_H_

#include "operators/update/update_op_base.h"

namespace dragon {

template <class Context>
class RMSPropUpdateOp final : public UpdateOpBase<Context> {
 public:
    RMSPropUpdateOp(const OperatorDef& op_def, Workspace* ws) 
        : UpdateOpBase<Context>(op_def, ws),
          eps(param("eps")),
          decay(param("decay")) {}

    void ComputeRunWithFloat() override;

 protected:
    float lr, decay, eps;
    unique_ptr<Tensor> history;
    Tensor temp;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_UPDATE_RMSPROP_UPDATE_OP_H_