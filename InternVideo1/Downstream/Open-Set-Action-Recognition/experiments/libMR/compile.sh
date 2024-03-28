#!/bin/bash

echo "----- Removing previously compiled libmr.so -----\n"
rm -r build
rm *.model
rm libmr.so
rm *.dump
rm ../libmr.so

echo "----- Building and compiling libmr ------- \n"
python setup.py build_ext -i
# cp libmr.so ../

# echo "----- Completed Compiling libmr -------- \n"
# echo "Now trying python -c \"import libmr\""
# python test_libmr.py
# echo "----- Compiling Done. Now import *.so file in your application -----\n"
