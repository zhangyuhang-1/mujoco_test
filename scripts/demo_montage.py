import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
import h5py
from PIL import Image,ImageDraw
import numpy as np

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('path'); p.add_argument('--output',default='outputs/demo/montage.jpg'); a=p.parse_args()
    phases=[(0.,'Initial'),(3.8,'Descend'),(5.2,'Grasp and lift'),(6.6,'Lifted'),(8.6,'Transfer'),(10.3,'Place'),(11.6,'Release'),(12.9,'Retreat / success')]
    canvas=Image.new('RGB',(4*256,2*280),'#f0f0f0')
    draw=ImageDraw.Draw(canvas)
    with h5py.File(a.path,'r') as f:
        indices=f['images/observation_index'][:]
        times=f['observations/time'][:][indices]
        for i,(time,label) in enumerate(phases):
            index=int(np.argmin(abs(times-time)))
            canvas.paste(Image.fromarray(f['images/overview'][index]),((i%4)*256,(i//4)*280))
            draw.text(((i%4)*256+5,(i//4)*280+260),f'{label}: {times[index]:.2f}s',fill='#202020')
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True); canvas.save(output)
    print(output)
