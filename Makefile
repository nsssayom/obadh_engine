.PHONY: all build test bench clean run-tokenizer

all: build

build:
	cargo build

test:
	cargo test

bench:
	cargo bench --bench hot_path

run-tokenizer:
	cargo run --features cli --bin tokenizer

tokenizer-test: build
	@echo "Testing tokenizer with some examples:"
	@echo "ndr" | cargo run --features cli --bin tokenizer
	@echo "kaby" | cargo run --features cli --bin tokenizer
	@echo "biSw" | cargo run --features cli --bin tokenizer

clean:
	cargo clean 
