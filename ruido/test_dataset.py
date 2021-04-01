from ruido.cc_dataset_mpi import CCDataset
# from ruido.cc_dataset_newnewnew import CCDataset
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

dat = CCDataset("/home/lermert/Desktop/CDMX/observations/datasets/G.UNM.00.HNZ--G.UNM.00.HNZ.pcc.windows.h5")#"testdata2.h5")

dat.data_to_memory()
if rank == 0:
    print(dat.dataset[0].data.shape)
print(dat)
if rank == 0:
    dat.dataset[0].lose_allzero_windows()
print(dat)
dat.filter_data(f_hp=1.0, f_lp=2.0)