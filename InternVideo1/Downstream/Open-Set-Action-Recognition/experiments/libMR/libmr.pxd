cdef extern from "MetaRecognition.h":
    cdef struct svm_node_libsvm:
        int index
        double value