binary=2584
all:
	g++ -std=c++11 -O3 -g -Wall -fmessage-length=0 -o $(binary) $(binary).cpp
	cp $(binary) ~/tcg
	chmod 755 $(binary)
	chmod +x ~/tcg/$(binary)
clean:
	rm $(binary)
	rm ~/tcg/$(binary)