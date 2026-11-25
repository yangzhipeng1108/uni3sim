// CPU-only inspection of public CANN operator registration. No ACL/NPU calls.
// Loads one explicitly named op library into a task-local registry and records
// callback identities without constructing a guessed compile-info layout.
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "base/registry/op_impl_space_registry_v2.h"

static void quoted(const char *value) {
  std::cout << '"';
  if (value != nullptr) {
    for (const unsigned char byte : std::string(value)) {
      if (byte == '"' || byte == '\\') std::cout << '\\' << byte;
      else if (byte < 32) {
        std::cout << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                  << unsigned(byte) << std::dec;
      } else std::cout << byte;
    }
  }
  std::cout << '"';
}

template <typename Function>
static bool callback(const char *name, Function function) {
  std::cout << '"' << name << "\":{";
  void *address = reinterpret_cast<void *>(function);
  std::cout << "\"present\":" << (address != nullptr ? "true" : "false");
  Dl_info info{};
  const bool resolved = address != nullptr && dladdr(address, &info) != 0;
  std::cout << ",\"dladdr_resolved\":" << (resolved ? "true" : "false");
  if (resolved) {
    std::cout << ",\"library\":"; quoted(info.dli_fname);
    std::cout << ",\"symbol\":"; quoted(info.dli_sname);
  }
  std::cout << '}';
  return resolved;
}

int main(int argc, char **argv) {
  if (argc != 2 || argv[1][0] != '/') {
    std::cerr << "usage: cast_registry_capture /absolute/path/liboptiling.so\n";
    return 2;
  }
  // Public AddSoToRegistry retains the implementation holder. Keep the local
  // registry alive until every callback identity has been printed.
  gert::OpImplSpaceRegistryV2 registry;
  gert::OppSoDesc description({ge::AscendString(argv[1])},
                            ge::AscendString("midlayer_cast_registry_capture"));
  const auto status = registry.AddSoToRegistry(description);
  if (status != ge::GRAPH_SUCCESS) {
    std::cerr << "public AddSoToRegistry failed: " << status << '\n';
    return 3;
  }
  std::cout << "{\"schema_version\":1,\"CPU_only\":true,"
               "\"duration_recorded\":false,\"operator_library\":";
  quoted(argv[1]);
  std::cout << ",\"registrations\":[";
  bool first = true;
  for (const char *key : {"DefaultImpl", "Cast"}) {
    if (!first) std::cout << ',';
    first = false;
    const auto *implementation = registry.GetOpImpl(key);
    std::cout << "{\"key\":"; quoted(key);
    std::cout << ",\"registered\":" << (implementation != nullptr ? "true" : "false");
    if (implementation != nullptr) {
      std::cout << ",\"max_tiling_data_size\":" << implementation->max_tiling_data_size
                << ",\"callbacks\":{";
      callback("tiling", implementation->tiling); std::cout << ',';
      callback("tiling_parse", implementation->tiling_parse); std::cout << ',';
      callback("compile_info_creator", implementation->compile_info_creator); std::cout << ',';
      callback("compile_info_deleter", implementation->compile_info_deleter);
      std::cout << '}';
    }
    std::cout << '}';
  }
  std::cout << "]}\n";
  return 0;
}
