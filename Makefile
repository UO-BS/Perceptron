CXX=g++
CXXFLAGS= -g -Wall

ifeq ($(OS), Windows_NT)
	RM=del
	ChangeOSPath = $(subst /,\,$1) 
else
	ifeq ($(shell uname), Linux)
		RM = rm -f
		ChangeOSPath = $(subst ,.exe,$1)
	endif
endif

SRCDIR=src
BUILDDIR=bin

SRCS=$(SRCDIR)/main.cpp $(SRCDIR)/Layer.cpp $(SRCDIR)/NeuralNetwork.cpp $(SRCDIR)/Neuron.cpp
OBJS=$(patsubst $(SRCDIR)/%.cpp,$(BUILDDIR)/%.o,$(SRCS))

all: $(BUILDDIR)/perceptron ;

clean:
	$(RM) $(call ChangeOSPath, $(BUILDDIR)/perceptron.exe)
	$(RM) $(call ChangeOSPath, $(OBJS))
	
$(BUILDDIR)/perceptron: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
