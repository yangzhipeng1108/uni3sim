#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "register/tilingdata_base.h"

namespace {

struct StructRecord {
  std::string path;
  std::string class_name;
  std::size_t offset;
  std::size_t size;
  std::size_t field_count;
};

struct FieldRecord {
  std::string path;
  std::string dtype;
  std::size_t offset;
  std::size_t count;
  std::size_t size;
};

struct OpaqueRecord {
  std::string path;
  std::string class_name;
  std::size_t offset;
  std::size_t size;
};

std::size_t DtypeSize(const std::string& dtype) {
  if (dtype == "bool" || dtype == "int8_t" || dtype == "uint8_t") {
    return 1;
  }
  if (dtype == "int16_t" || dtype == "uint16_t") {
    return 2;
  }
  if (dtype == "int32_t" || dtype == "uint32_t" || dtype == "float") {
    return 4;
  }
  if (dtype == "int64_t" || dtype == "uint64_t" || dtype == "double") {
    return 8;
  }
  throw std::runtime_error("unsupported tiling field dtype: " + dtype);
}

std::string RequireText(const char* value, const std::string& context) {
  if (value == nullptr || value[0] == '\0') {
    throw std::runtime_error(context + " must be non-empty");
  }
  std::string text(value);
  if (text.find('\t') != std::string::npos ||
      text.find('\n') != std::string::npos ||
      text.find('\r') != std::string::npos) {
    throw std::runtime_error(context + " contains a record delimiter");
  }
  return text;
}

std::shared_ptr<optiling::TilingDef> CreateTilingData(
    const std::string& registration_key) {
  auto instance = optiling::CTilingDataClassFactory::GetInstance()
                      .CreateTilingDataInstance(registration_key.c_str());
  if (instance == nullptr) {
    throw std::runtime_error("unregistered tiling-data key: " +
                             registration_key);
  }
  return instance;
}

void ReflectStruct(optiling::TilingDef* definition, std::size_t base_offset,
                   const std::string& path, std::size_t depth,
                   std::vector<StructRecord>* structs,
                   std::vector<FieldRecord>* fields,
                   std::vector<OpaqueRecord>* opaque,
                   bool allow_opaque) {
  if (definition == nullptr || structs == nullptr || fields == nullptr ||
      opaque == nullptr) {
    throw std::runtime_error("tiling reflection received a null object");
  }
  if (depth > 32) {
    throw std::runtime_error("tiling struct nesting exceeds 32 levels");
  }
  const std::string class_name = RequireText(
      definition->GetTilingClassName(), "tiling class name");
  const std::vector<optiling::FieldInfo> field_info =
      definition->GetFieldInfo();
  const std::size_t declared_size = definition->GetDataSize();
  structs->push_back(
      {path, class_name, base_offset, declared_size, field_info.size()});

  std::size_t relative_offset = 0;
  for (const auto& field : field_info) {
    const std::string field_name = RequireText(field.name_, "tiling field name");
    const std::string field_path =
        path == "$" ? field_name : path + "." + field_name;
    const std::string class_type = RequireText(
        field.classType_, "tiling field class type");
    std::size_t field_size = 0;
    if (class_type == "0" || class_type == "1") {
      const std::string dtype = RequireText(field.dtype_, "tiling field dtype");
      const std::size_t count = class_type == "0" ? 1 : field.arrSize_;
      if (count == 0) {
        throw std::runtime_error("zero-length tiling array: " + field_path);
      }
      field_size = DtypeSize(dtype) * count;
      fields->push_back(
          {field_path, dtype, base_offset + relative_offset, count, field_size});
    } else if (class_type == "2") {
      const std::string struct_type = RequireText(
          field.structType_, "nested tiling struct type");
      field_size = field.structSize_;
      std::shared_ptr<optiling::TilingDef> nested;
      try {
        nested = CreateTilingData(struct_type + "Op");
      } catch (const std::runtime_error&) {
        if (!allow_opaque) {
          throw;
        }
        opaque->push_back(
            {field_path, struct_type, base_offset + relative_offset,
             field_size});
      }
      if (nested != nullptr) {
        if (nested->GetDataSize() != field_size) {
          throw std::runtime_error(
              "nested tiling struct size mismatch for " + field_path);
        }
        ReflectStruct(nested.get(), base_offset + relative_offset, field_path,
                      depth + 1, structs, fields, opaque, allow_opaque);
      }
    } else {
      throw std::runtime_error("unsupported tiling field class type " +
                               class_type + " for " + field_path);
    }
    if (relative_offset > declared_size ||
        field_size > declared_size - relative_offset) {
      throw std::runtime_error("tiling field exceeds declared struct: " +
                               field_path);
    }
    relative_offset += field_size;
  }
  if (relative_offset != declared_size) {
    throw std::runtime_error("tiling fields do not cover declared struct " +
                             class_name);
  }
}

int Run(const std::string& library_path, const std::string& registration_key,
        bool allow_opaque) {
  void* handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (handle == nullptr) {
    const char* error = dlerror();
    throw std::runtime_error(
        "cannot load op-host library: " +
        std::string(error == nullptr ? "unknown dlopen error" : error));
  }

  auto root = CreateTilingData(registration_key);
  std::vector<StructRecord> structs;
  std::vector<FieldRecord> fields;
  std::vector<OpaqueRecord> opaque;
  ReflectStruct(root.get(), 0, "$", 0, &structs, &fields, &opaque,
                allow_opaque);

  std::cout << "SCHEMA\t1\n";
  std::cout << "ROOT\t" << registration_key << '\t'
            << RequireText(root->GetTilingClassName(), "root tiling class name")
            << '\t' << root->GetDataSize() << '\t' << structs.size() << '\t'
            << fields.size() << '\n';
  for (const auto& record : structs) {
    std::cout << "STRUCT\t" << record.path << '\t' << record.class_name << '\t'
              << record.offset << '\t' << record.size << '\t'
              << record.field_count << '\n';
  }
  for (const auto& record : fields) {
    std::cout << "FIELD\t" << record.path << '\t' << record.dtype << '\t'
              << record.offset << '\t' << record.count << '\t' << record.size
              << '\n';
  }
  for (const auto& record : opaque) {
    std::cout << "OPAQUE\t" << record.path << '\t' << record.class_name << '\t'
              << record.offset << '\t' << record.size << '\n';
  }
  if (!std::cout.good()) {
    throw std::runtime_error("failed to write tiling reflection output");
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::cerr << "usage: " << argv[0]
              << " <op-host-library> <registration-key> [--allow-opaque]\n";
    return 2;
  }
  const bool allow_opaque = argc == 4;
  if (allow_opaque && std::string(argv[3]) != "--allow-opaque") {
    std::cerr << "unknown tiling reflection option: " << argv[3] << '\n';
    return 2;
  }
  try {
    return Run(argv[1], argv[2], allow_opaque);
  } catch (const std::exception& error) {
    std::cerr << "tiling reflection failed: " << error.what() << '\n';
    return 1;
  }
}
