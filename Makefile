OFX_ROOT := vendor/openfx
BUILD := build
BUNDLE := $(BUILD)/DJIReframe.ofx.bundle
INSTALL_DIR := /usr/OFX/Plugins

CXXFLAGS := -std=c++17 -O3 -ffast-math -fPIC -Wall -Wextra -fvisibility=hidden \
            -I$(OFX_ROOT)/include -I$(OFX_ROOT)/Support/include \
            -DOFX_EXTENSIONS_NUKE -DOFX_EXTENSIONS_RESOLVE
LDFLAGS := -shared -lpthread

SUPPORT_SRC := $(wildcard $(OFX_ROOT)/Support/Library/ofxs*.cpp)
SRC := src/DJIReframe.cpp $(SUPPORT_SRC)
OBJ := $(patsubst %.cpp,$(BUILD)/obj/%.o,$(SRC))

all: $(BUNDLE)

$(BUILD)/obj/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUNDLE): $(OBJ)
	@mkdir -p $(BUNDLE)/Contents/Linux-x86-64
	$(CXX) $(OBJ) $(LDFLAGS) -o $(BUNDLE)/Contents/Linux-x86-64/DJIReframe.ofx
	@cp -f resources/*.png $(BUNDLE)/Contents/Resources/ 2>/dev/null || true

install: $(BUNDLE)
	sudo mkdir -p $(INSTALL_DIR)
	sudo rm -rf $(INSTALL_DIR)/DJIReframe.ofx.bundle
	sudo cp -r $(BUNDLE) $(INSTALL_DIR)/

clean:
	rm -rf $(BUILD)

.PHONY: all install clean
