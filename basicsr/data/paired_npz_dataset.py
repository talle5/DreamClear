import os
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import three_paths_from_folder, paired_paths_from_lmdb
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import torch
import numpy as np

@DATASET_REGISTRY.register(suffix='basicsr')
class PairedNpzDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedNpzDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        # mean and std for normalizing the input images
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder, self.npz_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_npz']
        self.filename_tmpl = opt['filename_tmpl'] if 'filename_tmpl' in opt else '{}'

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info' in self.opt and self.opt['meta_info'] is not None:
            # disk backend with meta_info
            # Each line in the meta_info describes the relative path to an image
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip() for line in fin]
            self.paths = []
            for path in paths:
                gt_path, lq_path = path.split(', ')
                gt_path = os.path.join(self.gt_folder, gt_path)
                lq_path = os.path.join(self.lq_folder, lq_path)
                self.paths.append(dict([('gt_path', gt_path), ('lq_path', lq_path)]))
        else:
            # disk backend
            # it will scan the whole folder to get meta info
            # it will be time-consuming for folders with too many files. It is recommended using an extra meta txt file
            self.paths = three_paths_from_folder([self.lq_folder, self.gt_folder, self.npz_folder], ['lq', 'gt', 'npz'], self.filename_tmpl)

        if 'num_pic' in self.opt:
            self.paths = self.paths[:self.opt['num_pic']]
        if 'phase' not in self.opt:
            self.opt['phase'] = 'test'


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)


        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        data_info = {'img_hw': torch.tensor([1024., 1024.], dtype=torch.float32), 'aspect_ratio': torch.tensor(1.)}

        npz_path = self.paths[index]['npz_path']
        txt_info = np.load(npz_path)
        txt_fea = torch.from_numpy(txt_info['caption_feature'])
        attention_mask = torch.ones(1, 1, txt_fea.shape[1])
        if 'attention_mask' in txt_info.keys():
            attention_mask = torch.from_numpy(txt_info['attention_mask'])[None]

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'npz_path': npz_path,'txt_fea': txt_fea, 'mask': attention_mask, 'data_info': data_info}

    def __len__(self):
        return len(self.paths)