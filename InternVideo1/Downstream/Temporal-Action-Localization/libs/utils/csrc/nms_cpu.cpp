#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <vector>

// 1D NMS (CPU) helper functions, ported from
// https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/pytorch/nms.cpp

using namespace at;

#define CHECK_CPU(x) \
  TORCH_CHECK(!x.device().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CPU_INPUT(x) \
  CHECK_CPU(x);            \
  CHECK_CONTIGUOUS(x)

Tensor nms_1d_cpu(Tensor segs, Tensor scores, float iou_threshold) {
  if (segs.numel() == 0) {
    return at::empty({0}, segs.options().dtype(at::kLong));
  }
  auto x1_t = segs.select(1, 0).contiguous();
  auto x2_t = segs.select(1, 1).contiguous();

  Tensor areas_t = x2_t - x1_t + 1e-6;

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto nsegs = segs.size(0);
  Tensor select_t = at::ones({nsegs}, segs.options().dtype(at::kBool));

  auto select = select_t.data_ptr<bool>();
  auto order = order_t.data_ptr<int64_t>();
  auto x1 = x1_t.data_ptr<float>();
  auto x2 = x2_t.data_ptr<float>();
  auto areas = areas_t.data_ptr<float>();

  for (int64_t _i = 0; _i < nsegs; _i++) {
    if (select[_i] == false) continue;
    auto i = order[_i];
    auto ix1 = x1[i];
    auto ix2 = x2[i];
    auto iarea = areas[i];

    for (int64_t _j = _i + 1; _j < nsegs; _j++) {
      if (select[_j] == false) continue;
      auto j = order[_j];
      auto xx1 = std::max(ix1, x1[j]);
      auto xx2 = std::min(ix2, x2[j]);

      auto inter = std::max(0.f, xx2 - xx1);
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr >= iou_threshold) select[_j] = false;
    }
  }
  return order_t.masked_select(select_t);
}

Tensor nms_1d(Tensor segs, Tensor scores, float iou_threshold) {
  CHECK_CPU_INPUT(segs);
  CHECK_CPU_INPUT(scores);
  return nms_1d_cpu(segs, scores, iou_threshold);

}

Tensor softnms_1d_cpu(Tensor segs, Tensor scores, Tensor dets, float iou_threshold,
                      float sigma, float min_score, int method) {
  if (segs.numel() == 0) {
    return at::empty({0}, segs.options().dtype(at::kLong));
  }

  auto x1_t = segs.select(1, 0).contiguous();
  auto x2_t = segs.select(1, 1).contiguous();
  auto scores_t = scores.clone();

  Tensor areas_t = x2_t - x1_t + 1e-6;

  auto nsegs = segs.size(0);
  auto x1 = x1_t.data_ptr<float>();
  auto x2 = x2_t.data_ptr<float>();
  auto sc = scores_t.data_ptr<float>();
  auto areas = areas_t.data_ptr<float>();
  auto de = dets.data_ptr<float>();

  int64_t pos = 0;
  Tensor inds_t = at::arange(nsegs, segs.options().dtype(at::kLong));
  auto inds = inds_t.data_ptr<int64_t>();

  for (int64_t i = 0; i < nsegs; i++) {
    auto max_score = sc[i];
    auto max_pos = i;

    // get seg with max score
    pos = i + 1;
    while (pos < nsegs) {
      if (max_score < sc[pos]) {
        max_score = sc[pos];
        max_pos = pos;
      }
      pos = pos + 1;
    }
    // swap the current seg (i) and the seg with max score (max_pos)
    auto ix1 = de[i * 3 + 0] = x1[max_pos];
    auto ix2 = de[i * 3 + 1] = x2[max_pos];
    auto iscore = de[i * 3 + 2] = sc[max_pos];
    auto iarea = areas[max_pos];
    auto iind = inds[max_pos];

    x1[max_pos] = x1[i];
    x2[max_pos] = x2[i];
    sc[max_pos] = sc[i];
    areas[max_pos] = areas[i];
    inds[max_pos] = inds[i];

    x1[i] = ix1;
    x2[i] = ix2;
    sc[i] = iscore;
    areas[i] = iarea;
    inds[i] = iind;

    // reset pos
    pos = i + 1;
    while (pos < nsegs) {
      auto xx1 = std::max(ix1, x1[pos]);
      auto xx2 = std::min(ix2, x2[pos]);

      auto inter = std::max(0.f, xx2 - xx1);
      auto ovr = inter / (iarea + areas[pos] - inter);

      float weight = 1.;
      if (method == 0) {
        // vanilla nms
        if (ovr >= iou_threshold) weight = 0;
      } else if (method == 1) {
        // linear
        if (ovr >= iou_threshold) weight = 1 - ovr;
      } else if (method == 2) {
        // gaussian
        weight = std::exp(-(ovr * ovr) / sigma);
      }
      sc[pos] *= weight;

      // if the score falls below threshold, discard the segment by
      // swapping with last seg update N
      if (sc[pos] < min_score) {
        x1[pos] = x1[nsegs - 1];
        x2[pos] = x2[nsegs - 1];
        sc[pos] = sc[nsegs - 1];
        areas[pos] = areas[nsegs - 1];
        inds[pos] = inds[nsegs - 1];
        nsegs = nsegs - 1;
        pos = pos - 1;
      }

      pos = pos + 1;
    }
  }
  return inds_t.slice(0, 0, nsegs);
}

Tensor softnms_1d(Tensor segs, Tensor scores, Tensor dets, float iou_threshold,
                  float sigma, float min_score, int method) {
  // softnms is not implemented on GPU
  CHECK_CPU_INPUT(segs)
  CHECK_CPU_INPUT(scores)
  CHECK_CPU_INPUT(dets)
  return softnms_1d_cpu(segs, scores, dets, iou_threshold, sigma, min_score, method);
}

// bind to torch interface
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "nms", &nms_1d, "nms (CPU) ",
    py::arg("segs"), py::arg("scores"), py::arg("iou_threshold")
  );
  m.def(
    "softnms", &softnms_1d, "softnms (CPU) ",
    py::arg("segs"), py::arg("scores"), py::arg("dets"), py::arg("iou_threshold"),
    py::arg("sigma"), py::arg("min_score"), py::arg("method")
  );
}
