EXENAME = main
OBJS = main.o ann_classifier.o 

CXX = clang++
CXXFLAGS = $(CS225) -std=c++1y -stdlib=libc++ -c -g -O0 -Wall -Wextra -pedantic
LD = clang++
LDFLAGS = -std=c++1y -stdlib=libc++ -lc++abi -lm

$(EXENAME) : $(OBJS)
	$(LD) $(OBJS) $(LDFLAGS) -o $(EXENAME)

main.o : output_msg main.cpp Classifier/ann_classifier.h
	$(CXX) $(CXXFLAGS) main.cpp

ann_classifier.o : Classifier/ann_classifier.cpp Classifier/ann_classifier.h Classifier/classifier.h
	$(CXX) $(CXXFLAGS) Classifier/ann_classifier.cpp

clean :
	-rm -f *.o $(EXENAME) test_

# Custom Clang version enforcement Makefile rule:
ccred=$(shell echo -e "\033[0;31m")
ccyellow=$(shell echo -e "\033[0;33m")
ccend=$(shell echo -e "\033[0m")

IS_EWS=$(shell hostname | grep "ews.illinois.edu")
IS_CORRECT_CLANG=$(shell clang -v 2>&1 | grep "version 6")
ifneq ($(strip $(IS_EWS)),)
ifeq ($(strip $(IS_CORRECT_CLANG)),)
CLANG_VERSION_MSG = $(error $(ccred) On EWS, please run 'module load llvm/6.0.1' first when running CS225 assignments. $(ccend))
endif
else
CLANG_VERSION_MSG = $(warning $(ccyellow) Looks like you are not on EWS. Be sure to test on EWS before the deadline. $(ccend))
endif

output_msg: ; $(CLANG_VERSION_MSG)

# Be sure to add output_msg as dependency target for your `intro` binary

tests: output_msg ann_test.o ann_classifier.o 
	$(LD) ann_test.o ann_classifier.o $(LDFLAGS) -o tests

ann_test.o : Test/ann_test.cpp Test/catch/catch.hpp Classifier/ann_classifier.h
	$(CXX) $(CXXFLAGS) Test/ann_test.cpp

.PHONY: output_msg