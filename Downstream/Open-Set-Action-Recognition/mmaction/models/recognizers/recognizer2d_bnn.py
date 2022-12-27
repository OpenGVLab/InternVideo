from ..registry import RECOGNIZERS
from .recognizer2d import Recognizer2D


@RECOGNIZERS.register_module()
class Recognizer2DBNN(Recognizer2D):
    """2D recognizer model framework."""

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        losses = dict()

        x = self.extract_feat(imgs)

        if hasattr(self, 'debias_head'):
            loss_debias = self.debias_head(x, num_segs=num_segs, target=labels.squeeze(), **kwargs)
            losses.update(loss_debias)

        if hasattr(self, 'neck'):
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x, labels.squeeze())
            x = x.squeeze(2)
            num_segs = 1

        outputs = self.cls_head(x, num_segs, npass=self.train_cfg['npass'], testing=False)
        # parse the outputs
        cls_score = outputs['pred_mean']
        gt_labels = labels.squeeze()
        loss_dict = self.cls_head.bnn_loss(cls_score, gt_labels, outputs, beta=self.train_cfg['loss_weight'], **kwargs)
        losses.update(loss_dict)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]

        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        losses = dict()

        x = self.extract_feat(imgs)
        if hasattr(self, 'neck'):
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, loss_aux = self.neck(x)
            x = x.squeeze(2)
            losses.update(loss_aux)
            num_segs = 1

        # When using `TSNHead` or `TPNHead`, shape is [batch_size, num_classes]
        # When using `TSMHead`, shape is [batch_size * num_crops, num_classes]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop/MultiGroupCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        outputs = self.cls_head(x, num_segs, npass=self.test_cfg['npass'], testing=True)
        cls_score = outputs['pred_mean']
        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)

        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs).cpu().numpy()

    def forward_dummy(self, imgs):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)
        outputs = (self.cls_head(x, num_segs, npass=self.test_cfg['npass'], testing=True), )
        outs = (outputs['pred_mean'], )
        return outs
