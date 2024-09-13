import argparse, torch, os, json, uuid
import constants, models
from galois_common import gcutils
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class CustomDataset(Dataset):
    def __init__(self, image_folder, data_path, image_processor, tokenizer) -> None:
        super().__init__()
        self.image_folder = image_folder
        self.data_dict = [json.loads(q) for q in open(os.path.expanduser(data_path), 'r')]
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __len__(self): return len(self.data_dict)

    def __getitem__(self, k):
        source = self.data_dict[k]
        image = self._process_image(source)
        conv = self._process_conversation(source)
        data_dict = self._preprocess(image, conv)
        data_dict['question_id'] = source['question_id']
        data_dict['text'] = source['text']
        return data_dict

    def _preprocess(self, image, conversations):
        input_ids = self._make_token_ids(conversations)
        return {'image': image, 'input_ids': torch.tensor(input_ids)}

    def _process_image(self, source):
        if 'image' not in source: return None
        path = os.path.join(self.image_folder, source['image'])
        image = Image.open(path)
        processed = self.image_processor.preprocess(image, return_tensors='np')['pixel_values'][0]
        return processed.transpose(1, 2, 0)

    def _process_conversation(self, source):
        return [
            {'role': 'USER', 'value': constants.DEFAULT_IMAGE_TOKEN + '\n' + source['text']},
            {'role': 'ASSISTANT', 'value': None}
        ]

    def _make_token_ids(self, conversations):
        input_ids = self.tokenizer(constants.SYSTEM_MESSAGE).input_ids
        for x in conversations:
            prefix = self.tokenizer(f"{x['role']}:", add_special_tokens=False).input_ids
            tokens = self.tokenizer(f"{x['value']}", add_special_tokens=False).input_ids
            input_ids.extend(prefix + tokens)
            if x['role'] == 'ASSISTANT':
                input_ids.append(self.tokenizer.eos_token_id)
        return input_ids


def collate_fn(batch):
    return batch[0]
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes

# DataLoader
def create_data_loader(data_folder, image_processor, tokenizer, batch_size=1, num_workers=4):
    assert batch_size == 1, 'batch_size must be 1'
    image_folder = os.path.join(data_folder, 'test2015')
    question_path = os.path.join(data_folder, 'llava_vqav2_mscoco_test2015.jsonl')
    dataset = CustomDataset(image_folder, question_path, image_processor, tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

def eval_model(args):
    # Model
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
    model = models.load(args.model_path, torch.bfloat16)
    model.cuda()
    answers_file = os.path.join(args.output_folder, 'answer.jsonl')
    gcutils.ensure_folder(answers_file)
    ans_file = open(answers_file, 'w')
    data_loader = create_data_loader(args.input_folder, model.vision.processor, model.language.tokenizer)

    for line in tqdm(data_loader):
        idx = line['question_id']
        cur_prompt = line['text']
        input_ids = line['input_ids'].to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids[None, :],
                images=[line['image']],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
        outputs = model.language.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        ans_id = str(uuid.uuid4())
        ans_file.write(json.dumps({'question_id': idx,
                                   'prompt': cur_prompt,
                                   'text': outputs,
                                   'answer_id': ans_id,
                                   'model_id': 'clip_llm',
                                   'metadata': {}}) + '\n')
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='data/checkpoints/finetune/complete')
    parser.add_argument('--input-folder', type=str, default='data/llm/eval/vqav2')
    parser.add_argument('--output-folder', type=str, default='data/eval/vqa2')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    args = parser.parse_args()
    eval_model(args)
