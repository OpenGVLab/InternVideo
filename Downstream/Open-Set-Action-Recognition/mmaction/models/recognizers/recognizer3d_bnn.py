from ..registry import RECOGNIZERS
from .recognizer3d import Recognizer3D


@RECOGNIZERS.register_module()
class Recognizer3DBNN(Recognizer3D):
    """3D recognizer model framework."""

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        losses = dict()

        x = self.extract_feat(imgs)
        if hasattr(self, 'neck'):
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)

        outputs = self.cls_head(x, npass=self.train_cfg['npass'], testing=False)
        # parse the outputs
        cls_score = outputs['pred_mean']
        gt_labels = labels.squeeze()
        loss_dict = self.cls_head.bnn_loss(cls_score, gt_labels, outputs, beta=self.train_cfg['loss_weight'], **kwargs)
        losses.update(loss_dict)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        x = self.extract_feat(imgs)
        if hasattr(self, 'neck'):
            x, _ = self.neck(x)

        outputs = self.cls_head(x, npass=self.test_cfg['npass'], testing=True)
        cls_score = outputs['pred_mean']
        cls_score = self.average_clip(cls_score, num_segs)

        return cls_score


    def forward_dummy(self, imgs):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        x = self.extract_feat(imgs)
        outputs = self.cls_head(x, npass=self.test_cfg['npass'], testing=True)
        outs = (outputs['pred_mean'], )
        return outs
