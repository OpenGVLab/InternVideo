#include "ROIAlign3d.h"
#include "ROIPool3d.h"
#include "SoftmaxFocalLoss.h"
#include "SigmoidFocalLoss.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_align_3d_forward",&ROIAlign3d_forward, "ROIAlign3d_forward");
  m.def("roi_align_3d_backward",&ROIAlign3d_backward, "ROIAlign3d_backward");
  m.def("roi_pool_3d_forward", &ROIPool3d_forward, "ROIPool3d_forward");
  m.def("roi_pool_3d_backward", &ROIPool3d_backward, "ROIPool3d_backward");
  m.def("sigmoid_focalloss_forward", &SigmoidFocalLoss_forward, "SigmoidFocalLoss_forward");
  m.def("sigmoid_focalloss_backward", &SigmoidFocalLoss_backward, "SigmoidFocalLoss_backward");
  m.def("softmax_focalloss_forward", &SoftmaxFocalLoss_forward, "SoftmaxFocalLoss_forward");
  m.def("softmax_focalloss_backward", &SoftmaxFocalLoss_backward, "SoftmaxFocalLoss_backward");
}
