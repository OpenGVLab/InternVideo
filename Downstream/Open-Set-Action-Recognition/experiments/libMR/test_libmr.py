import scipy as sp
import sys, os
try:
    import libmr
    print("Imported libmr succesfully")
except ImportError:
    print("Cannot import libmr")
    sys.exit()

import pickle
svm_data = {}
svm_data["labels"] = [1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1 , -1, -1, -1, -1, -1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1 , -1, -1, -1, -1, -1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1 , -1, -1, -1, -1, -1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1 , -1, -1, -1, -1, -1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1 , -1, -1, -1, -1, -1]
svm_data["scores"] = sp.randn(100).tolist()
fit_data = sp.rand(3)
def main():

    mr = libmr.MR()
    datasize = len(svm_data["scores"])
    mr.fit_svm(svm_data, datasize, 1, 1, 1, 10)
    print(fit_data)
    print(mr.w_score_vector(fit_data))
    mr.mr_save("meta_rec.model")
    datadump = {}
    datadump = {"data": fit_data}

    f = open("data.dump", "w")
    pickle.dump(datadump, f)
    f.close()
    print(dir(mr))


if __name__ == "__main__":
    main()
