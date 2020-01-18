from jma_pytorch_dataset import *
from utils import AverageMeter, Logger
from criteria_precip import *

# --------------------------
# Test
# --------------------------

def test_epoch(test_loader,model,opt):
    print('Testing for the model')
    
    # initialize
    RMSE_all = np.empty((0,opt.tdim_use),float)
    Xpast_all = np.empty((0,opt.tdim_use),float)
    Xtrue_all = np.empty((0,opt.tdim_use),float)
    Xmodel_all = np.empty((0,opt.tdim_use),float)
    
    for i_batch, sample_batched in enumerate(test_loader):
        past = sample_batched['past'].float()
        target = sample_batched['future'].float()
        # Forward
        # use the lates data from past as prediction
        output = np.zeros(past.shape)
        for n in range(target.shape[1]):
            output[:,n,:] = past[:,-1]
            
        # concat all prediction data
        Xtrue = target.squeeze()
        Xmodel = output.squeeze()
        #
        Xpast_all = np.append(Xpast_all,past.squeeze(),axis=0)
        Xtrue_all = np.append(Xtrue_all,Xtrue,axis=0)
        Xmodel_all = np.append(Xmodel_all,Xmodel,axis=0)

    # prep csv
    tpred = (np.arange(opt.tdim_use)+1.0)*5.0 # in minutes
    RMSE = np.sqrt(np.mean((Xtrue_all-Xmodel_all)**2,axis=0))
    # import pdb; pdb.set_trace()
    df_out = pd.DataFrame({'tpred_min':tpred,
                       'RMSE':RMSE})
    # apply eval metric by rain level
    levels = np.arange(-10,220,10)
    for i in range(len(levels)-1):
        low = levels[i]
        high = levels[i+1]
        id_range = (Xpast_all[:,-1] > low) * (Xpast_all[:,-1] <= high)
        print("range: ",low,high,"number of samples: ",np.sum(id_range))
        # calc rmse
        xt = Xtrue_all[id_range,:]
        xm = Xmodel_all[id_range,:]
        # RMSE along "samples" axis and keep time dim
        rr = np.sqrt(np.mean((xt-xm)**2,axis=0))
        vname = "RMSE_%d_%d" % (low,high)
        df_out[vname] = rr

    # save evaluated metric as csv file
    df_out.to_csv(os.path.join(opt.result_path,
                           'test_evaluation_predtime.csv'), float_format='%.3f')

    
