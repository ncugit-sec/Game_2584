binary=2584
all: compile
	cp $(binary) ~/tcg
	chmod 755 $(binary)
	chmod +x ~/tcg/$(binary)
compile:
	g++ -std=c++11 -O3 -g -Wall -fmessage-length=0 -o $(binary) $(binary).cpp
clean:
	rm $(binary)
	rm ~/tcg/$(binary)