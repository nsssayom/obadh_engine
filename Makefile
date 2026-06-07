.PHONY: all build test bench clean run-tokenizer

all: build

build:
	cargo build

test:
	cargo test

bench:
	cargo bench --bench hot_path

run-tokenizer:
	cargo run --bin tokenizer

tokenizer-test: build
	@echo "Testing tokenizer with some examples:"
	@echo "ndr" | cargo run --bin tokenizer
	@echo "kaby" | cargo run --bin tokenizer
	@echo "biSw" | cargo run --bin tokenizer

clean:
	cargo clean 
