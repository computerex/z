# Harness Go Build Configuration

# Go parameters
GO := go
GOBIN := bin
LDFLAGS := -s -w

# Version
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
LDFLAGS += -X main.Version=$(VERSION)

# Output names
BINARY := harness
WINDOWS_BINARY := harness.exe
DARWIN_BINARY := harness-darwin
LINUX_BINARY := harness-linux

.PHONY: all clean build build-windows build-darwin build-linux build-all install test lint

# Default target
all: build

# Build for current platform
build:
	$(GO) build -ldflags="$(LDFLAGS)" -o $(GOBIN)/$(BINARY)

# Build for Windows
build-windows:
	GOOS=windows GOARCH=amd64 $(GO) build -ldflags="$(LDFLAGS)" -o $(GOBIN)/$(WINDOWS_BINARY)

# Build for Darwin (macOS)
build-darwin:
	GOOS=darwin GOARCH=amd64 $(GO) build -ldflags="$(LDFLAGS)" -o $(GOBIN)/$(DARWIN_BINARY)
	GOOS=darwin GOARCH=arm64 $(GO) build -ldflags="$(LDFLAGS)" -o $(GOBIN)/$(DARWIN_BINARY)-arm64

# Build for Linux
build-linux:
	GOOS=linux GOARCH=amd64 $(GO) build -ldflags="$(LDFLAGS)" -o $(GOBIN)/$(LINUX_BINARY)

# Build all platforms
build-all: build-windows build-darwin build-linux
	@echo "All binaries built successfully:"
	@ls -la $(GOBIN)/*

# Clean build artifacts
clean:
	rm -rf $(GOBIN)/*
	rm -f coverage.out

# Install to system
install:
	$(GO) install -ldflags="$(LDFLAGS)"

# Run tests
test:
	$(GO) test -v ./...

# Run with coverage
test-coverage:
	$(GO) test -coverprofile=coverage.out ./...
	$(GO) tool cover -html=coverage.out -o coverage.html

# Run linter
lint:
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run ./...; \
	else \
		echo "golangci-lint not installed, skipping..."; \
	fi

# Download dependencies
deps:
	$(GO) mod download
	$(GO) mod tidy

# Format code
fmt:
	$(GO) fmt ./...

# Show dependency tree
deps-tree:
	@go mod graph | head -50

# Create release zip
release-zip:
	@mkdir -p release
	@cd release && zip -r ../harness-$(VERSION)-windows-amd64.zip $(WINDOWS_BINARY) README.md LICENSE
	@cd release && zip -r ../harness-$(VERSION)-darwin-amd64.zip $(DARWIN_BINARY) README.md LICENSE
	@cd release && zip -r ../harness-$(VERSION)-linux-amd64.zip $(LINUX_BINARY) README.md LICENSE
	@echo "Release zips created in release/"

# Show build info
info:
	@echo "Version: $(VERSION)"
	@echo "Go version: $(shell go version)"
	@echo "OS/ARCH: $(shell go env GOOS)/$(shell go env GOARCH)"