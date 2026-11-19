CXX ?= g++
CPPFLAGS += -Iinclude $(YAML_CFLAGS)
CXXFLAGS ?= -O2 -g -Wall -Wextra -Wpedantic
YAML_LIBS ?= -lyaml
BUILD_DIR ?= build$(if $(filter 1,$(HIP)),/hip)$(if $(filter 1,$(RDMA)),/rdma)

ifeq ($(RDMA),1)
CPPFLAGS += -DEF_HAVE_RDMA $(RDMA_CFLAGS)
RDMA_LIBS ?= -lrdmacm -libverbs
endif

sources := $(wildcard src/*.cpp)
objects := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(sources))

ifeq ($(HIP),1)
CPPFLAGS += -DEF_HAVE_HIP
ROCM_PATH ?= /opt/rocm
HIPCC ?= $(ROCM_PATH)/bin/hipcc
HIP_ARCHS ?= gfx1100
HIP_LIBS ?= -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64
hip_object := $(BUILD_DIR)/hip_expert.o
objects += $(hip_object)
endif

.PHONY: all test clean
all: $(BUILD_DIR)/expert-fabric

$(BUILD_DIR)/expert-fabric: $(objects)
	$(CXX) $(LDFLAGS) $^ -pthread $(YAML_LIBS) $(RDMA_LIBS) $(HIP_LIBS) $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: src/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++20 -pthread -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: src/%.hip
	@mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CPPFLAGS) $(CXXFLAGS) $(addprefix --offload-arch=,$(HIP_ARCHS)) -std=c++20 -MMD -MP -c $< -o $@

$(BUILD_DIR)/expert-concurrency-test: tests/expert_concurrency.cpp $(BUILD_DIR)/expert.o $(hip_object)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++20 -pthread $(LDFLAGS) $^ $(HIP_LIBS) $(LDLIBS) -o $@

$(BUILD_DIR)/ngram-test: tests/ngram.cpp $(BUILD_DIR)/ngram.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++20 -pthread $(LDFLAGS) $^ $(LDLIBS) -o $@

test: all $(BUILD_DIR)/expert-concurrency-test $(BUILD_DIR)/ngram-test
	$(BUILD_DIR)/expert-concurrency-test
	$(BUILD_DIR)/ngram-test
	python3 tests/smoke.py $(BUILD_DIR)/expert-fabric $(if $(filter 1,$(HIP)),--hip)

clean:
	rm -f $(objects) $(objects:.o=.d) $(BUILD_DIR)/expert-fabric $(BUILD_DIR)/expert-concurrency-test $(BUILD_DIR)/ngram-test

-include $(objects:.o=.d)
