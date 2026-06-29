#!/bin/bash

# Exit on any error
set -e

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Show an error message and exit
error() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    exit 1
}

# Show a success message
success() {
    echo -e "${GREEN}SUCCESS: $1${NC}"
}

# Show an info message
info() {
    echo -e "${BLUE}INFO: $1${NC}"
}

# Show a warning message
warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

# Check for required tools
check_requirements() {
    info "Checking requirements..."
    
    # Check for Rust
    if ! command -v rustc &> /dev/null; then
        error "Rust is not installed. Please install it from https://rustup.rs/"
    fi
    
    # Check for wasm-pack
    if ! command -v wasm-pack &> /dev/null; then
        info "wasm-pack is not installed. Installing..."
        cargo install wasm-pack || error "Failed to install wasm-pack"
    fi
    
    # Check for npm
    if ! command -v npm &> /dev/null; then
        error "Node.js/npm is not installed. Please install it from https://nodejs.org/"
    fi

    # Check for Binaryen wasm-opt. wasm-pack's bundled optimizer can lag behind
    # current Rust WASM features, so the build uses the system binary with
    # explicit bulk-memory support.
    if ! command -v wasm-opt &> /dev/null; then
        warning "Binaryen wasm-opt is not installed. WASM optimization will be skipped."
    fi
    
    # Change to www directory and install npm dependencies if needed
    if [ -d "www" ] && [ -f "www/package.json" ]; then
        cd www
        if [ ! -d "node_modules" ]; then
            info "Installing npm dependencies..."
            npm install || error "Failed to install npm dependencies"
        fi
        cd ..
    fi
    
    success "All requirements satisfied"
}

# Clean function to remove temporary files
clean() {
    info "Cleaning build artifacts..."
    
    # Remove target directory
    if [ -d "target" ]; then
        rm -rf target
        info "Removed 'target' directory"
    fi
    
    # Remove wasm-pack generated files
    if [ -d "pkg" ]; then
        rm -rf pkg
        info "Removed 'pkg' directory"
    fi
    
    success "Clean complete!"
    return 0
}

# Copy and fix paths in index.html for docs directory
fix_index_paths() {
    local src="$1"
    local dst="$2"
    
    info "Fixing paths in index.html for GitHub Pages..."
    
    # Copy the file first
    cp "$src" "$dst" || error "Failed to copy index.html"
    
    # No path fixes needed as all resources use relative paths already
    # If needed, this is where we would adjust paths for GitHub Pages
}

# Build the WASM package and setup web files
build_wasm() {
    info "Building WASM Package..."
    
    # Build WebAssembly package
    wasm-pack build --target web --out-dir pkg --no-opt --profile wasm-release || error "Failed to build WebAssembly package"
    optimize_wasm
    
    # Check for www directory
    if [ ! -d "www" ]; then
        error "www directory does not exist. Please create it first."
    fi
    
    # Copy WASM files to www/js
    cd www
    mkdir -p js
    cp ../pkg/*.js js/ || error "Failed to copy JS files"
    cp ../pkg/*.wasm js/ || error "Failed to copy WASM files"
    
    # Fix import paths in JS files
    for jsfile in js/*.js; do
        sed -i.bak 's|import.meta.url, \"../pkg/|import.meta.url, \"|g' "$jsfile" && rm -f "$jsfile.bak"
    done
    cd ..
    
    success "WASM build complete!"
    return 0
}

# Optimize generated WASM with Binaryen.
optimize_wasm() {
    if ! command -v wasm-opt &> /dev/null; then
        warning "Skipping WASM optimization because wasm-opt is unavailable"
        return 0
    fi

    for wasmfile in pkg/*.wasm; do
        [ -e "$wasmfile" ] || continue
        local optimized="${wasmfile}.opt"
        wasm-opt --enable-bulk-memory --enable-nontrapping-float-to-int -O "$wasmfile" -o "$optimized" || error "Failed to optimize $wasmfile"
        mv "$optimized" "$wasmfile" || error "Failed to replace optimized WASM file"
    done
}

# Build Tailwind CSS
build_css() {
    info "Building CSS..."
    
    cd www
    npm run build || error "Failed to build CSS"
    cd ..
    
    success "CSS build complete!"
    return 0
}

# Build for distribution (GitHub Pages)
build_dist() {
    info "Building project for distribution (GitHub Pages)..."
    
    # First build the WASM package
    info "Building WebAssembly package..."
    wasm-pack build --target web --out-dir pkg --no-opt --profile wasm-release || error "Failed to build WebAssembly package"
    optimize_wasm
    
    # Change to www directory
    cd www || error "Failed to change to www directory"
    
    # Copy WASM files to js directory
    info "Copying WebAssembly files..."
    mkdir -p js
    cp ../pkg/*.js js/ || error "Failed to copy JS files"
    cp ../pkg/*.wasm js/ || error "Failed to copy WASM files"
    
    # Fix import paths in JS files
    info "Fixing module paths in JavaScript files..."
    for jsfile in js/*.js; do
        sed -i.bak 's|import.meta.url, \"../pkg/|import.meta.url, \"|g' "$jsfile" && rm -f "$jsfile.bak"
    done
    
    # Build CSS
    info "Building CSS..."
    npm run build || error "Failed to build CSS"
    
    # Return to project root
    cd ..
    
    # Clean and prepare docs directory
    info "Preparing docs directory for GitHub Pages..."
    if [ -d "docs" ]; then
        # Simply clean the directory - docs/ is only for GitHub Pages
        rm -rf docs/* || error "Failed to clean docs directory"
    else
        mkdir -p docs || error "Failed to create docs directory"
    fi
    
    # Copy index.html with path fixes
    fix_index_paths "www/index.html" "docs/index.html"
    
    # Copy all necessary files to docs directory
    info "Copying files to docs directory..."
    mkdir -p docs/css
    cp www/css/styles.css docs/css/ || error "Failed to copy CSS files"
    mkdir -p docs/js
    cp www/js/*.js docs/js/ || error "Failed to copy JS files"
    cp www/js/*.wasm docs/js/ || error "Failed to copy WASM files"
    
    # Check for other assets that might need to be copied
    if [ -d "www/assets" ]; then
        mkdir -p docs/assets
        cp -r www/assets/* docs/assets/ || error "Failed to copy assets directory"
    fi

    if [ -d "www/vendor" ]; then
        mkdir -p docs/vendor
        cp -r www/vendor/* docs/vendor/ || error "Failed to copy vendor directory"
    fi
    
    # Success message
    success "Distribution build complete! Files are ready in the docs directory."
    info "You can now commit the docs directory for GitHub Pages deployment."
    info "Your site will be available at: https://sayom.me/obadh_engine/"
    return 0
}

# Serve the web application
serve() {
    info "Starting development server..."
    
    # Change to www directory
    cd www
    
    # Setup signal handling
    # This is a cleaner approach than using background processes with trap
    exec npm run serve
}

# Development mode with watch
dev() {
    info "Starting development environment with watch..."
    
    # Change to www directory
    cd www
    
    # Run npm dev command that handles CSS watch and server
    exec npm run dev
}

# Build everything and start the server
start() {
    info "Building and starting the server..."
    
    cd www
    npm run build-wasm && npm run build && npm run serve
    cd ..
}

# Build the native Rust binary
build_bin() {
    info "Building native Rust binary..."
    
    # Build in release mode for optimization
    cargo build --release --bin obadh || error "Failed to build native binary"
    
    success "Native binary built successfully at: target/release/obadh"
    info "You can install it to your system with: cargo install --path ."
    return 0
}

# Build everything (bin, wasm, css, dist)
build_all() {
    info "Building everything (native binary, WASM, CSS, and distribution files)..."
    
    # First clean everything
    clean
    
    # Build the native binary
    build_bin
    
    # Build for distribution (this includes building WASM and CSS)
    build_dist
    
    success "All components built successfully!"
    info "Native binary is available at: target/release/obadh"
    info "Web files are ready in the docs/ directory for GitHub Pages."
    info ""
    info "To serve the web interface locally:"
    info "cd docs && python -m http.server 8000"
    info "Then visit: http://localhost:8000"
    info ""
    info "Or you can just deploy to GitHub Pages:"
    info "git add docs/"
    info "git commit -m \"Update deployment files\""
    info "git push"
    
    return 0
}

# Display the help information
show_help() {
    echo "Obadh Engine Build Tool"
    echo "======================="
    echo "Usage:"
    echo "  ./build.sh bin      # Build the native Rust binary (bin/obadh)"
    echo "  ./build.sh wasm     # Build the WASM package"
    echo "  ./build.sh css      # Build Tailwind CSS"
    echo "  ./build.sh serve    # Start the development server only"
    echo "  ./build.sh dev      # Start development mode with file watching"
    echo "  ./build.sh start    # Build everything and start the server"
    echo "  ./build.sh dist     # Build for distribution (GitHub Pages)"
    echo "  ./build.sh clean    # Clean up build artifacts"
    echo "  ./build.sh all      # Build everything (bin, wasm, css, dist)"
    echo ""
    echo "Note: Using 'dev' or 'serve' is the recommended way for development."
    echo "      Use 'dist' to prepare files for GitHub Pages deployment."
    echo "      Use 'bin' to build the command-line tool."
    echo "      Use 'all' to build everything for production and distribution."
}

# Main execution
case "$1" in
    "clean")
        clean
        ;;
    "wasm")
        check_requirements
        build_wasm
        ;;
    "css")
        check_requirements
        build_css
        ;;
    "serve")
        check_requirements
        serve
        ;;
    "dev")
        check_requirements
        dev
        ;;
    "dist")
        check_requirements
        build_dist
        ;;
    "start")
        check_requirements
        clean && build_wasm && build_css && serve
        ;;
    "bin")
        check_requirements
        build_bin
        ;;
    "all")
        check_requirements
        build_all
        ;;
    *)
        show_help
        ;;
esac

exit $?
