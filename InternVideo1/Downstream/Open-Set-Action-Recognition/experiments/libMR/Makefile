CXX= g++
SRC= MetaRecognition.cpp weibull.c

libmr: $(SRC) weibull.h malloc.h MetaRecognition.h
	$(CXX) -o libmr $(SRC) -I.

clean:
	rm -f *~ *.o libmr