all:
	g++ -std=c++11 -O3 -g -Wall -fmessage-length=0 -o 2048 2048.cpp
	cp 2048 ~/tcg
	chmod 755 2048
clean:
	rm 2048
	rm ~/tcg/2048