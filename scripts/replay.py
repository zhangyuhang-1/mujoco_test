import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
from dualstack.data.episodes import replay_video

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('path'); p.add_argument('--output',default='outputs/replay.mp4'); p.add_argument('--camera',default='overview'); a=p.parse_args()
    print(replay_video(a.path,a.output,a.camera))
