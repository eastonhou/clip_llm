import torch, transformers
import numpy as np
from torch import nn
from galois_runtime import imutils, grutils, grmodel
from galois_common import gcutils

class VisionModelBase(grmodel.ModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, images):
        def _insert_zeros(values, default):
            itr = iter(values)
            values[:] = [default if x is None else next(itr) for x in images]
        _images = [x for x in images if x is not None]
        _outputs = self._forward(_images)
        _lengths = [x.shape[0] for x in _outputs['hidden_states']]
        _insert_zeros(_outputs['hidden_states'], torch.zeros(0, 1024, device=self.device))
        _insert_zeros(_outputs['position'], np.zeros((0, 2), dtype=int))
        _insert_zeros(_lengths, 0)
        return {
            'hidden_states': torch.cat(_outputs['hidden_states'], 0),
            'position': np.concatenate(_outputs['position'], 0),
            'lengths': _lengths
        }

    def _forward(self, images):
        images, positions, scl_shapes, ali_shapes = self.preprocess(images, self.patch_size)
        batch_patches = [imutils.chunk_image(x, self.chunk_size, self.overlap_size, 0) for x in images]
        all_patches = np.array([x['image'] for x in gcutils.flatten(batch_patches)])
        embeddings = self.encode(all_patches)
        start = 0
        outputs = []
        for patches, shape in zip(batch_patches, ali_shapes):
            _embeddings = imutils.merge_chunk2d(embeddings[start:start + len(patches)], patches, self.patch_size, self.overlap_size)
            _embeddings = _embeddings[:shape[0] // self.patch_size, :shape[1] // self.patch_size]
            output = _embeddings.reshape(-1, _embeddings.shape[-1])
            outputs.append(output)
            start += len(patches)
        return {
            'hidden_states': outputs,
            'position': positions,
        }

    def preprocess(self, images, patch_size):
        ref_size = self.chunk_size * self.max_chunks_per_dim - self.overlap_size * (self.max_chunks_per_dim - 1)
        scales = [min(ref_size / max(x.shape[:2]), 1) for x in images]
        images = [imutils.scale_image(x, scale) for x, scale in zip(images, scales)]
        scl_shapes = [x.shape[:2] for x in images]
        ali_shapes = [(imutils.align(np.array(x), self.patch_size)) for x in scl_shapes]
        aligned = [self.auto_pad(x) for x in images]
        #aligned = [imutils.align_image(x, patch_size) for x in images]
        positions = []
        for s in ali_shapes:
            _height, _width = s
            _ys, _xs = np.mgrid[:_height // patch_size, :_width // patch_size]
            _pos = np.stack([_ys, _xs], -1)
            _pos = _pos.reshape(-1, 2)
            positions.append(_pos)
        return aligned, positions, scl_shapes, ali_shapes

    def auto_pad(self, image):
        def _blocks(v):
            num = np.ceil((v - self.chunk_size) / (self.chunk_size - self.overlap_size)) + 1
            num = max(int(num), 1)
            return num
        def _size(blocks):
            return blocks * self.chunk_size - (blocks - 1) * self.overlap_size
        block_y, block_x = _blocks(image.shape[0]), _blocks(image.shape[1])
        height, width = _size(block_y), _size(block_x)
        return imutils.pad_image(image, (height, width))

    @property
    def chunk_size(self): raise NotImplementedError()
    @property
    def patch_size(self): raise NotImplementedError()
    @property
    def hidden_size(self): raise NotImplementedError()
    @property
    def overlap_size(self): raise NotImplementedError()
    @property
    def max_chunks_per_dim(self): raise NotImplementedError()

    def encode(self, images): # returns tensor of shape (bn, height, width, dimension)
        raise NotImplementedError()

class VisionModel(VisionModelBase):
    def __init__(self, model_name, dtype) -> None:
        super().__init__()
        self.processor = transformers.CLIPImageProcessor.from_pretrained(model_name, do_resize=False, do_center_crop=False)
        self.module = transformers.CLIPVisionModel.from_pretrained(model_name, torch_dtype=dtype)
        self.module.requires_grad_(False)

    @property
    def hidden_size(self): return self.module.config.hidden_size
    @property
    def chunk_size(self): return self.module.config.image_size
    @property
    def patch_size(self): return self.module.config.patch_size
    @property
    def overlap_size(self): return 28
    @property
    def max_chunks_per_dim(self): return 3

    @torch.no_grad()
    def encode(self, images):
        if images.size == 0: return torch.zeros((0, 0, 0, self.hidden_size), dtype=self.module.dtype)
        inputs = self.tensor(images.transpose(0, 3, 1, 2)).to(self.module.dtype)
        outputs = self.module(inputs, output_hidden_states=True)
        features = outputs.hidden_states[-2][:, 1:]
        bs, wh, dim = features.shape
        embeddings = features.view(bs, int(wh**0.5), int(wh**0.5), dim)
        return embeddings

    def save(self, folder):
        self.processor.save_pretrained(folder)
        self.module.save_pretrained(folder)
