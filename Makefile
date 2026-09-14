OFX_ROOT := vendor/openfx
BUILD := build
BUNDLE := $(BUILD)/DJIReframe.ofx.bundle
INSTALL_DIR := /usr/OFX/Plugins

CL_INC := $(firstword $(wildcard /usr/include/CL/cl.h) $(wildcard /opt/cuda/include/CL/cl.h))
CL_FLAGS := $(if $(CL_INC),-I$(patsubst %/CL/cl.h,%,$(CL_INC)))

CXXFLAGS := -std=c++17 -O3 -ffast-math -fPIC -Wall -Wextra -fvisibility=hidden \
            -I$(OFX_ROOT)/include -I$(OFX_ROOT)/Support/include -Isrc $(CL_FLAGS) \
            -DOFX_EXTENSIONS_NUKE -DOFX_EXTENSIONS_RESOLVE
LDFLAGS := -shared -lpthread -lOpenCL -lGL

SUPPORT_SRC := $(wildcard $(OFX_ROOT)/Support/Library/ofxs*.cpp)
SRC := src/DJIReframe.cpp src/ReframeOpenCL.cpp $(SUPPORT_SRC)
OBJ := $(patsubst %.cpp,$(BUILD)/obj/%.o,$(SRC))

all: $(BUNDLE)

# The kernel is kept as real .cl so it stays diffable against the extracted original.
src/reframe_kernel.inc: src/reframe_kernel.cl
	@sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/"/' -e 's/$$/\\n"/' $< > $@

$(BUILD)/obj/src/DJIReframe.o: src/reframe_kernel.inc
$(BUILD)/obj/src/ReframeOpenCL.o: src/reframe_kernel.inc

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
	rm -rf $(BUILD) src/reframe_kernel.inc

.PHONY: all install clean
