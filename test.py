from options import test_options
from dataloader import data_loader
from model import create_model
from itertools import islice
from torch.cuda import Event
import torch
from tqdm import tqdm
import os

if __name__ == '__main__':
    # get testing options
    opt = test_options.TestOptions().parse()
    # creat a dataset
    dataset = data_loader.dataloader(opt)
    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    model = create_model(opt)
    model.eval()
    # create a visualizer

    times = []
    total_iters = opt.how_many if opt.how_many != 0 else len(dataset)
    iterator = islice(dataset, opt.how_many) if opt.how_many != 0 else dataset

    for i, data in enumerate(tqdm(iterator, total=total_iters), 1):

        model.set_input(data)
        t=model.test()

        times.append(t)
    if times:
        with open(os.path.join(opt.results_dir, 'result.txt'), 'w') as f:
            avg_batch = sum(times) / len(times)
            avg_image = avg_batch / opt.batchSize
            print(f'平均Inference用時:{avg_image:.4f}s')
            fps = 1 / avg_image  # 每秒可處理張數
            print(f'InferenceFPS:{fps:.2f}張/s')
            f.write(f'Inference Time:{avg_image} FPS:{fps}\n')
