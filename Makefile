all:
	g++ -g -lboost_filesystem -lboost_system `pkg-config --cflags opencv` `pkg-config --libs opencv` recognition.cpp
clean:
	rm -r a.out
