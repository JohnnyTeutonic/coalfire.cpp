#pragma once

/**
 * @file gguf_writer.hpp
 * @brief The GGUF v3 byte writer, standalone (standard library only).
 *
 * One writer for every GGUF the paperkiln family writes from C++:
 * coalfire.cpp's export_to_gguf (src/gguf_export.cpp) and paperkiln's
 * export_gguf_llama / _gpt2 / _paperkiln (paperkiln's src/gguf.cpp, which
 * takes this file from its vendored third_party/transformer_cpp). The
 * exporters decide what goes into a file; this header decides only how it
 * is laid out:
 *
 *   header        magic "GGUF", version 3, tensor count (u64), key count (u64)
 *   metadata      each key: string key, u32 value type, value; in the order
 *                 the write_metadata_* calls were made
 *   tensor infos  each tensor: string name, u32 n_dims, n_dims u64 dims
 *                 fastest-varying first (ne0 = the contiguous axis), u32
 *                 ggml type, u64 offset from the data section's start
 *   padding       zeros up to the next multiple of GGUF_DEFAULT_ALIGNMENT:
 *                 the data section starts there, as readers mmap it
 *                 (2026-07-13: without this padding every tensor was read
 *                 byte-shifted and the logits came out NaN; EXPORT_NOTES.md)
 *   data          each tensor at its offset, every offset a multiple of the
 *                 alignment, zero padding between tensors; the last tensor
 *                 is padded too only under set_pad_final_tensor(true)
 *
 * Strings are a u64 byte count and the bytes, without a terminator. All
 * values are little-endian (the host order of every supported target). The
 * writer writes no key by itself: an exporter writes general.alignment (32)
 * like any other key.
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gguf_export {

// GGUF constants
constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF" in little-endian
constexpr uint32_t GGUF_VERSION = 3;
constexpr uint64_t GGUF_DEFAULT_ALIGNMENT = 32;

// GGML tensor types (ggml.h numbering) that the exporters write.
enum class GGMLType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    BF16 = 30
};

// GGUF value types for metadata
enum class GGUFValueType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12
};

// A ggml type's storage: blocks of block_elems values in block_bytes bytes
// (one value per block for the float types).
struct GGMLTypeTraits {
    uint64_t block_elems;
    uint64_t block_bytes;
};

inline GGMLTypeTraits ggml_type_traits(GGMLType type) {
    switch (type) {
        case GGMLType::F32: return {1, 4};
        case GGMLType::F16: return {1, 2};
        case GGMLType::BF16: return {1, 2};
        case GGMLType::Q4_0: return {32, 18};
        case GGMLType::Q4_1: return {32, 20};
        case GGMLType::Q5_0: return {32, 22};
        case GGMLType::Q5_1: return {32, 24};
        case GGMLType::Q8_0: return {32, 34};
        case GGMLType::Q4_K: return {256, 144};
        case GGMLType::Q5_K: return {256, 176};
        case GGMLType::Q6_K: return {256, 210};
    }
    throw std::invalid_argument("gguf_writer: unknown ggml type " + std::to_string(static_cast<uint32_t>(type)));
}

// The bytes `num_elements` values of `type` occupy; throws unless they fill
// whole blocks.
inline uint64_t ggml_nbytes(GGMLType type, uint64_t num_elements) {
    const GGMLTypeTraits t = ggml_type_traits(type);
    if (num_elements % t.block_elems != 0)
        throw std::invalid_argument("gguf_writer: " + std::to_string(num_elements) +
                                    " values are not whole blocks of " + std::to_string(t.block_elems) +
                                    " for ggml type " + std::to_string(static_cast<uint32_t>(type)));
    return num_elements / t.block_elems * t.block_bytes;
}

/**
 * @brief One tensor to write.
 *
 * shape is in row-major (numpy) order and is written REVERSED, so the last
 * entry is ne0, the contiguous axis: an [out][in] buffer passed as
 * {out, in} lands as ne = {in, out}. data points at the tensor's bytes,
 * ggml_nbytes(type, num_elements) of them (num_elements floats for F32, the
 * quantised blocks otherwise), and must stay valid until finalize().
 */
struct TensorExportInfo {
    std::string name;
    std::vector<uint64_t> shape;
    GGMLType type = GGMLType::F32;
    const void* data = nullptr;
    size_t num_elements = 0;
};

/**
 * @brief Buffers metadata and tensors, then writes the file in finalize().
 */
class GGUFWriter {
public:
    explicit GGUFWriter(std::string filename) : filename_(std::move(filename)) {}

    // Metadata, written in call order.
    void write_metadata_uint8(const std::string& key, uint8_t v) { scalar(key, GGUFValueType::UINT8, v); }
    void write_metadata_int8(const std::string& key, int8_t v) { scalar(key, GGUFValueType::INT8, v); }
    void write_metadata_uint16(const std::string& key, uint16_t v) { scalar(key, GGUFValueType::UINT16, v); }
    void write_metadata_int16(const std::string& key, int16_t v) { scalar(key, GGUFValueType::INT16, v); }
    void write_metadata_uint32(const std::string& key, uint32_t v) { scalar(key, GGUFValueType::UINT32, v); }
    void write_metadata_int32(const std::string& key, int32_t v) { scalar(key, GGUFValueType::INT32, v); }
    void write_metadata_uint64(const std::string& key, uint64_t v) { scalar(key, GGUFValueType::UINT64, v); }
    void write_metadata_int64(const std::string& key, int64_t v) { scalar(key, GGUFValueType::INT64, v); }
    void write_metadata_float32(const std::string& key, float v) { scalar(key, GGUFValueType::FLOAT32, v); }
    void write_metadata_float64(const std::string& key, double v) { scalar(key, GGUFValueType::FLOAT64, v); }
    void write_metadata_bool(const std::string& key, bool v) {
        scalar(key, GGUFValueType::BOOL, static_cast<uint8_t>(v ? 1 : 0));
    }
    void write_metadata_string(const std::string& key, const std::string& value) {
        std::vector<uint8_t> kv = begin_kv(key, GGUFValueType::STRING);
        put_string(kv, value);
        metadata_.push_back(std::move(kv));
    }
    void write_metadata_string_array(const std::string& key, const std::vector<std::string>& values) {
        std::vector<uint8_t> kv = begin_array(key, GGUFValueType::STRING, values.size());
        for (const auto& s : values) put_string(kv, s);
        metadata_.push_back(std::move(kv));
    }
    void write_metadata_float_array(const std::string& key, const std::vector<float>& values) {
        array(key, GGUFValueType::FLOAT32, values);
    }
    void write_metadata_int32_array(const std::string& key, const std::vector<int32_t>& values) {
        array(key, GGUFValueType::INT32, values);
    }
    void write_metadata_uint32_array(const std::string& key, const std::vector<uint32_t>& values) {
        array(key, GGUFValueType::UINT32, values);
    }

    // Tensors, written in call order. Throws if the shape does not hold
    // num_elements values or ne0 is not a whole number of the type's blocks.
    void add_tensor(const TensorExportInfo& tensor) {
        uint64_t n = 1;
        for (uint64_t d : tensor.shape) n *= d;
        if (tensor.shape.empty() || n != tensor.num_elements)
            throw std::invalid_argument("gguf_writer: " + tensor.name + ": shape holds " + std::to_string(n) +
                                        " values, num_elements is " + std::to_string(tensor.num_elements));
        if (tensor.shape.back() % ggml_type_traits(tensor.type).block_elems != 0)
            throw std::invalid_argument("gguf_writer: " + tensor.name + ": ne0 = " +
                                        std::to_string(tensor.shape.back()) + " is not whole blocks");
        if (!tensor.data && tensor.num_elements)
            throw std::invalid_argument("gguf_writer: " + tensor.name + ": no data");
        tensors_.push_back(tensor);
    }

    // Pad the last tensor's data to the alignment as well (llama.cpp's
    // quantiser does; a plain export ends at the last tensor's last byte).
    void set_pad_final_tensor(bool pad) { pad_final_tensor_ = pad; }

    size_t metadata_count() const { return metadata_.size(); }
    size_t tensor_count() const { return tensors_.size(); }

    // Write the file. False (with a message on stderr) if it cannot be
    // opened or written.
    bool finalize() {
        std::ofstream file(filename_, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "GGUF Export Error: Could not open file " << filename_ << std::endl;
            return false;
        }
        std::vector<uint8_t> head;
        put(head, GGUF_MAGIC);
        put(head, GGUF_VERSION);
        put(head, static_cast<uint64_t>(tensors_.size()));
        put(head, static_cast<uint64_t>(metadata_.size()));
        for (const auto& kv : metadata_) head.insert(head.end(), kv.begin(), kv.end());

        uint64_t offset = 0;
        for (const auto& t : tensors_) {
            put_string(head, t.name);
            put(head, static_cast<uint32_t>(t.shape.size()));
            for (size_t i = t.shape.size(); i-- > 0;) put(head, t.shape[i]);
            put(head, static_cast<uint32_t>(t.type));
            offset = align(offset);
            put(head, offset);
            offset += ggml_nbytes(t.type, t.num_elements);
        }
        head.resize(align(head.size()), 0);
        file.write(reinterpret_cast<const char*>(head.data()), static_cast<std::streamsize>(head.size()));

        static const char zeros[GGUF_DEFAULT_ALIGNMENT] = {};
        uint64_t at = 0;
        for (size_t i = 0; i < tensors_.size(); ++i) {
            const auto& t = tensors_[i];
            const uint64_t start = align(at);
            file.write(zeros, static_cast<std::streamsize>(start - at));
            const uint64_t nbytes = ggml_nbytes(t.type, t.num_elements);
            file.write(static_cast<const char*>(t.data), static_cast<std::streamsize>(nbytes));
            at = start + nbytes;
        }
        if (pad_final_tensor_) file.write(zeros, static_cast<std::streamsize>(align(at) - at));
        file.close();
        if (!file) {
            std::cerr << "GGUF Export Error: Could not write file " << filename_ << std::endl;
            return false;
        }
        std::cout << "GGUF Export: Successfully wrote " << tensors_.size() << " tensors to " << filename_
                  << std::endl;
        return true;
    }

private:
    std::string filename_;
    std::vector<std::vector<uint8_t>> metadata_;  // each key's bytes, as written
    std::vector<TensorExportInfo> tensors_;
    bool pad_final_tensor_ = false;

    static uint64_t align(uint64_t x) { return (x + GGUF_DEFAULT_ALIGNMENT - 1) & ~(GGUF_DEFAULT_ALIGNMENT - 1); }

    template <typename T>
    static void put(std::vector<uint8_t>& out, const T& v) {
        const auto* p = reinterpret_cast<const uint8_t*>(&v);
        out.insert(out.end(), p, p + sizeof(T));
    }
    static void put_string(std::vector<uint8_t>& out, const std::string& s) {
        put(out, static_cast<uint64_t>(s.size()));
        out.insert(out.end(), s.begin(), s.end());
    }
    static std::vector<uint8_t> begin_kv(const std::string& key, GGUFValueType type) {
        std::vector<uint8_t> kv;
        put_string(kv, key);
        put(kv, static_cast<uint32_t>(type));
        return kv;
    }
    static std::vector<uint8_t> begin_array(const std::string& key, GGUFValueType elem, size_t n) {
        std::vector<uint8_t> kv = begin_kv(key, GGUFValueType::ARRAY);
        put(kv, static_cast<uint32_t>(elem));
        put(kv, static_cast<uint64_t>(n));
        return kv;
    }
    template <typename T>
    void scalar(const std::string& key, GGUFValueType type, T v) {
        std::vector<uint8_t> kv = begin_kv(key, type);
        put(kv, v);
        metadata_.push_back(std::move(kv));
    }
    template <typename T>
    void array(const std::string& key, GGUFValueType elem, const std::vector<T>& values) {
        std::vector<uint8_t> kv = begin_array(key, elem, values.size());
        for (const T& v : values) put(kv, v);
        metadata_.push_back(std::move(kv));
    }
};

}  // namespace gguf_export
