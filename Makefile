CFLAGS = -Wall -Wextra -O3 -Ilib -march=native
CC = gcc

SRCS = $(wildcard src/*.c)
OBJS = $(patsubst src/%.c,bin/%.o,$(SRCS))
BIN = bin/main

.PHONY: all clean run

all: $(BIN)

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(BIN) -lm

# main.c has no matching header, so it gets its own rule with no .h prerequisite.
bin/main.o: src/main.c | bin
	$(CC) $(CFLAGS) -c $< -o $@

# Everything else is assumed to have a matching header in lib/.
bin/%.o: src/%.c lib/%.h | bin
	$(CC) $(CFLAGS) -c $< -o $@

bin:
	mkdir -p bin

run: $(BIN)
	./$(BIN)

clean:
	rm -f bin/*.o $(BIN)