from tqdm import tqdm
import logging
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
import json
from torch.utils.data import Dataset

from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         SegMapTransform, Numpy2Tensor)
from .utils import to_tensor, random_scale
from .extra_aug import ExtraAugmentation


class MMTSVDataset(Dataset):
    CLASSES = None

    def __init__(self,
                 data, split, version,
                 img_scale,
                 img_norm_cfg,
                 multiscale_mode='value',
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 with_semantic_seg=False,
                 seg_prefix=None,
                 seg_scale_factor=1,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False):

        from qd.qd_pytorch import TSVSplitProperty
        self.image_tsv = TSVSplitProperty(data, split, t=None, version=version)
        self.label_tsv = TSVSplitProperty(data, split, t='label', version=version)
        self.hw_tsv = TSVSplitProperty(data, split, t='hw')
        from qd.tsv_io import TSVDataset
        self.labelmap = TSVDataset(data).load_labelmap()
        self.label_to_idx = {l: i + 1 for i, l in enumerate(self.labelmap)}

        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not test_mode:
            self.valid_inds = self._filter_imgs()
        else:
            self.valid_inds = None

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # multi-scale mode (only applicable for multi-scale training)
        self.multiscale_mode = multiscale_mode
        assert multiscale_mode in ['value', 'range']

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        # with semantic segmentation (stuff) annotation or not
        self.with_seg = with_semantic_seg
        # prefix of semantic segmentation map path
        self.seg_prefix = seg_prefix
        # rescale factor for segmentation maps
        self.seg_scale_factor = seg_scale_factor
        # in test mode or not
        self.test_mode = test_mode

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.mask_transform = MaskTransform()
        self.seg_transform = SegMapTransform(self.size_divisor)
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

    def __len__(self):
        if self.valid_inds is not None:
            return len(self.valid_inds)
        else:
            return len(self.hw_tsv)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        if self.valid_inds is not None:
            idx = self.valid_inds[idx]
        _, str_rects = self.label_tsv[idx]
        rects = json.loads(str_rects)
        # make sure the width and height is at least 1. Note, we should put
        # this logic in the function of computing the valid_index or solve it
        # in the data preparation. we leave this logic here mainly for parity
        # check of this class with the built-in custom/coco dataset
        crowd_idx = [i for i, r in enumerate(rects) if r.get('iscrowd')
                and r['rect'][2] - r['rect'][0] >= 0
                and r['rect'][3] - r['rect'][1] >= 0]
        non_crowd_idx = [i for i, r in enumerate(rects) if not r.get('iscrowd')
                and r['rect'][2] - r['rect'][0] >= 0
                and r['rect'][3] - r['rect'][1] >= 0]

        bbox = np.array([rects[i]['rect'] for i in non_crowd_idx],
                dtype=np.float32)
        if len(bbox) == 0:
            bbox = np.zeros((0, 4), dtype=np.float32)
        labels = np.array([self.label_to_idx[rects[i]['class']] for i in non_crowd_idx],
                dtype=np.int64)

        bboxes_ignore = np.array([rects[i]['rect'] for i in crowd_idx],
                dtype=np.float32)
        if len(bboxes_ignore) == 0:
            bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        labels_ignore = np.array([self.label_to_idx[rects[i]['class']] for i
                in non_crowd_idx], dtype=np.int64)

        return {'bboxes': bbox,
                'labels': labels,
                'bboxes_ignore': bboxes_ignore,
                'labels_ignore': labels_ignore}

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        logging.info('filtering which image is valid')
        for i in tqdm(range(len(self.hw_tsv))):
            _, str_rects = self.label_tsv[i]
            if len(json.loads(str_rects)) == 0:
                continue
            _, str_hw = self.hw_tsv[i]
            h, w = [int(x) for x in str_hw.split(' ')]
            if min(h, w) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            h, w = self.read_hw(i)
            if 1. * w / h > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        # load image
        img = self.read_image(idx)
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes,
                                                       gt_labels)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()
        if self.with_seg:
            raise NotImplementedError()
            #gt_seg = mmcv.imread(
                #osp.join(self.seg_prefix, img_info['file_name'].replace(
                    #'jpg', 'png')),
                #flag='unchanged')
            #gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            #gt_seg = mmcv.imrescale(
                #gt_seg, self.seg_scale_factor, interpolation='nearest')
            #gt_seg = gt_seg[None, ...]
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           scale_factor, flip)
        h, w = self.read_hw(idx)
        ori_shape = (h, w, 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        #if self.with_seg:
            #data['gt_semantic_seg'] = DC(to_tensor(gt_seg), stack=True)
        data['id'] = self.read_key(idx)
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img = self.read_image(idx)
        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        img_h, img_w = self.read_hw(idx)

        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_h, img_w, 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data

    def read_image(self, idx):
        if self.valid_inds is not None:
            idx = self.valid_inds[idx]
        _, __, str_im = self.image_tsv[idx]
        from qd.qd_common import img_from_base64
        return img_from_base64(str_im)

    def read_hw(self, idx):
        if self.valid_inds is not None:
            idx = self.valid_inds[idx]
        _, str_hw = self.hw_tsv[idx]
        return [int(x) for x in str_hw.split(' ')]

    def read_key(self, idx):
        if self.valid_inds is not None:
            idx = self.valid_inds[idx]
        key, _ = self.hw_tsv[idx]
        return key

