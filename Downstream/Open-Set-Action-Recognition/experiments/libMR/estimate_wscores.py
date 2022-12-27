import os, sys
import scipy as sp
import libmr

def main():

    posscores = sp.asarray([0.245 ,  0.2632,  0.3233,  0.3573,  0.4014,  0.4055,  0.4212, 0.5677])
    test_distances = sp.asarray([ 0.05,  0.1 ,  0.25,  0.4 ,  0.75,  1.  ,  1.5 ,  2.])

    mr = libmr.MR()
    # since higher is worse and we want to fit the higher tail,
    # use fit_high()
    mr.fit_high(posscores, posscores.shape[0])
    wscores = mr.w_score_vector(test_distances)
    for i in range(wscores.shape[0]):
        print "%.2f %.2f %.2f" %(test_distances[i], wscores[i], mr.inv(wscores[i]))
    # wscores are the ones to be used in the equation
    # s_i * (1 - rho_i)
    print "Low wscore --> Low probability that the score is outlier i.e. sample IS NOT outlier"
    print "High wscore --> High probability that the score is outlier i.e. sample IS an outlier"
    print "posscores: ", posscores
    print "test_distances: ", test_distances
    print "wscores: ", wscores

if __name__ == "__main__":
    main()
