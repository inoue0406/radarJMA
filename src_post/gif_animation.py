#
# Create GIF animation from png files
#
import glob
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

from PIL import Image
        
if __name__ == '__main__':
    
    # read case name from command line
    argvs = sys.argv
    argc = len(argvs)

    if argc != 2:
        print('Usage: python gif_animation.py CASENAME')
        quit()

    case = argvs[1]
    #case = 'result_20190625_clstm_lrdecay07_ep20'
    #case = 'result_20190712_tr_clstm_flatsampled'
    pic_path = case + '/png/*dt00.png'

    # create pic save dir
    if not os.path.exists(case + '/gif'):
        os.mkdir(case + '/gif')

    for infile in sorted(glob.iglob(pic_path)):
        outgif = infile.replace('.h5_dt00.png','.gif')
        outgif = outgif.replace('png','gif')
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
        


