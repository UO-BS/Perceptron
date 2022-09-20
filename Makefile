CXX=g++
CXXFLAGS= -g -Wall

ifeq ($(OS), Windows_NT)
	RM=cmd /C del
	ChangeOSPath = $(subst /,\,$1) 
else
	ifeq ($(shell uname), Linux)
		RM = rm -f
		ChangeOSPath = $(subst ,.exe,$1)
	endif
endif

SRCDIR=src
OBJDIR=bin

SRCS=$(SRCDIR)/main.cpp $(SRCDIR)/Layer.cpp $(SRCDIR)/NeuralNetwork.cpp $(SRCDIR)/Neuron.cpp
OBJS=$(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))

all: $(OBJDIR)/perceptron ;

clean:
	$(RM) $(call ChangeOSPath,$(OBJDIR)/perceptron.exe)
	$(RM) $(call ChangeOSPath,$(OBJS))
	
$(OBJDIR)/perceptron: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJS): | $(OBJDIR)

$(OBJDIR):
	mkdir $(OBJDIR)