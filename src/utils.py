from pytubefix import YouTube
import torch
from tqdm import tqdm

def download_videos(path_links, path_videos):
    f = open(path_links, 'r')

    for link in tqdm(f):
        YouTube(link).streams.get_highest_resolution().download(path_videos)


def set_seed():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True




if __name__ == '__main__':
    download_videos("../videos/test_videos.txt", '../videos/originals')
