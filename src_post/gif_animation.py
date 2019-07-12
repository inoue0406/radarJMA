#
# Create GIF animation from png files
#
import glob

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

from PIL import Image
        
if __name__ == '__main__':

    pic_path = 'result_20190708_vclstm_modtst/png/*dt00.png'

    for infile in sorted(glob.iglob(pic_path)):
        outgif = infile.replace('dt00.png','.gif')
        ims = []
        fig=plt.figure(figsize=(16, 8))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        for n in range(6):
            in_dt = infile.replace('dt00','dt%02d' % n)
            print(infile)
            # read files
            img = Image.open(in_dt)
            im = plt.imshow(np.asarray(img))
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=300)
        plt.axis('off')
        plt.show(block=False)
        ani.save(outgif, writer="imagemagick")
        plt.close()
        


