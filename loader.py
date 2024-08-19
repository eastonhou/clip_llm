import torch, json, os
from torch.utils.data import Dataset, Sampler
from PIL import Image
from typing import Dict, Sequence
from dataclasses import dataclass
from constants import *

class SupervisedDataset(Dataset):
    def __init__(self, image_processor, tokenizer, image_folder, data_path) -> None:
        super().__init__()
        self.image_folder = image_folder
        self.data_dict = json.load(open(data_path, 'r'))
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __len__(self): return len(self.data_dict)

    def __getitem__(self, k) -> Dict[str, torch.Tensor]:
        source = self.data_dict[k]
        image = self._process_image(source)
        conversations = [self._process_conversation(x) for x in source['conversations']]
        data_dict = self._preprocess(image, conversations)
        return data_dict

    def _preprocess(self, image, conversations):
        input_ids, target_ids = self._make_token_ids(conversations)
        return {'image': image, 'input_ids': torch.tensor(input_ids), 'labels': torch.tensor(target_ids)}

    def _process_image(self, source):
        if 'image' not in source: return None
        path = os.path.join(self.image_folder, source['image'])
        image = Image.open(path)
        processed = self.image_processor.preprocess(image, return_tensors='np')['pixel_values'][0]
        return processed

    def _process_conversation(self, sentence):
        sentence = sentence.copy()
        roles = {'human': 'USER', 'gpt': 'ASSISTANT'}
        if DEFAULT_IMAGE_TOKEN in sentence['value']:
            sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            sentence['value'] = f"{DEFAULT_IMAGE_TOKEN}\n{sentence['value']}"
        sentence['role'] = roles[sentence.pop('from')]
        return sentence

    def _make_token_ids(self, conversations):
        input_ids = self.tokenizer(SYSTEM_MESSAGE).input_ids
        target_ids = [IGNORE_INDEX] * len(input_ids)
        for x in conversations:
            prefix = self.tokenizer(f"{x['role']}:", add_special_tokens=False).input_ids
            tokens = self.tokenizer(f"{x['value']}", add_special_tokens=False).input_ids
            input_ids.extend(prefix + tokens)
            if x['role'] == 'USER':
                target_ids.extend([IGNORE_INDEX] * len(prefix + tokens))
            else:
                target_ids.extend([IGNORE_INDEX] * len(prefix) + tokens)
                input_ids.append(self.tokenizer.eos_token_id)
                target_ids.append(self.tokenizer.eos_token_id)
        return input_ids, target_ids

@dataclass
class SupervisedDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [x['input_ids'] for x in instances]
        labels = [x['labels'] for x in instances]
        input_ids = self._pad_sequence(input_ids)
        labels = self._pad_sequence(labels)
        batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        batch['images'] = [x['image'].transpose(1, 2, 0) if x['image'] is not None else None for x in instances]
        return batch

    def _pad_sequence(self, sequences):
        padded = torch.nn.utils.rnn.pad_sequence(sequences, True, self.tokenizer.pad_token_id)
        padded = padded[:, :self.tokenizer.model_max_length]
        return padded

class LengthGroupedSampler(Sampler):
    def __init__(self, batch_size: int, world_size: int, datasource: Dataset):
        self.batch_size = batch_size
        self.world_size = world_size
        self.datasource = datasource
        self.indices = torch.randperm(len(self))

    def __len__(self): return len(self.datasource)

    def __iter__(self):
        return iter(self.indices)
