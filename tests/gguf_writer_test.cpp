// gguf_writer.hpp on its own: a small file written through it is compared
// byte for byte with the same file assembled here by hand from the GGUF v3
// specification, and the writer's refusals are checked.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf_writer.hpp"

namespace {

int failures = 0;
#define EXPECT(cond)                                                    \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                 \
        }                                                               \
    } while (0)

struct Bytes {
    std::vector<uint8_t> b;
    template <typename T>
    void raw(T v) {
        uint8_t p[sizeof(T)];
        std::memcpy(p, &v, sizeof(T));
        b.insert(b.end(), p, p + sizeof(T));
    }
    void str(const std::string& s) {
        raw<uint64_t>(s.size());
        b.insert(b.end(), s.begin(), s.end());
    }
    void key(const std::string& k, uint32_t type) {
        str(k);
        raw<uint32_t>(type);
    }
    void zeros_to(size_t multiple) {
        while (b.size() % multiple) b.push_back(0);
    }
};

std::vector<uint8_t> slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

std::string tmp_dir() {
    const char* t = std::getenv("TMPDIR");
#ifdef _WIN32
    if (!t) t = std::getenv("TEMP");
#endif
    return t ? std::string(t) : std::string("/tmp");
}

template <typename F>
bool throws(F f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    using namespace gguf_export;
    const std::string tmp = tmp_dir();

    // Block sizes (ggml.c's type_traits).
    EXPECT(ggml_nbytes(GGMLType::F32, 7) == 28);
    EXPECT(ggml_nbytes(GGMLType::F16, 7) == 14);
    EXPECT(ggml_nbytes(GGMLType::BF16, 7) == 14);
    EXPECT(ggml_nbytes(GGMLType::Q4_0, 64) == 36);
    EXPECT(ggml_nbytes(GGMLType::Q4_1, 64) == 40);
    EXPECT(ggml_nbytes(GGMLType::Q5_0, 64) == 44);
    EXPECT(ggml_nbytes(GGMLType::Q5_1, 64) == 48);
    EXPECT(ggml_nbytes(GGMLType::Q8_0, 64) == 68);
    EXPECT(ggml_nbytes(GGMLType::Q4_K, 512) == 288);
    EXPECT(ggml_nbytes(GGMLType::Q5_K, 512) == 352);
    EXPECT(ggml_nbytes(GGMLType::Q6_K, 512) == 420);
    EXPECT(throws([] { ggml_nbytes(GGMLType::Q8_0, 33); }));
    EXPECT(throws([] { ggml_type_traits(static_cast<GGMLType>(99)); }));

    // Three tensors whose sizes are not multiples of 32 bytes: a 1-D F32 of
    // 5 values (20 bytes), a 2-D F32 [3][2] (24 bytes, ne = {2, 3}) and one
    // Q8_0 block row [1][32] (34 bytes).
    const float a[5] = {1, 2, 3, 4, 5};
    const float m[6] = {0.5f, -0.5f, 1.5f, -1.5f, 2.5f, -2.5f};
    uint8_t q[34];
    for (int i = 0; i < 34; ++i) q[i] = static_cast<uint8_t>(i * 7 + 1);

    for (const bool pad_end : {false, true}) {
        const std::string path = tmp + "/gguf_writer_test.gguf";
        {
            GGUFWriter w(path);
            w.write_metadata_string("general.architecture", "llama");
            w.write_metadata_uint32("general.alignment", 32);
            w.write_metadata_uint8("t.u8", 200);
            w.write_metadata_int8("t.i8", -3);
            w.write_metadata_uint16("t.u16", 60000);
            w.write_metadata_int16("t.i16", -30000);
            w.write_metadata_int32("t.i32", -7);
            w.write_metadata_uint64("t.u64", 1ull << 40);
            w.write_metadata_int64("t.i64", -(1ll << 40));
            w.write_metadata_float32("t.f32", 0.25f);
            w.write_metadata_float64("t.f64", -0.125);
            w.write_metadata_bool("t.bool", true);
            w.write_metadata_string_array("t.strs", {"a", "", "xyz"});
            w.write_metadata_float_array("t.f32s", {1.0f, -2.0f});
            w.write_metadata_int32_array("t.i32s", {3, -4, 5});
            w.write_metadata_uint32_array("t.u32s", {});
            EXPECT(w.metadata_count() == 16);

            TensorExportInfo t1;
            t1.name = "vec";
            t1.shape = {5};
            t1.data = a;
            t1.num_elements = 5;
            w.add_tensor(t1);
            TensorExportInfo t2;
            t2.name = "mat";
            t2.shape = {3, 2};
            t2.data = m;
            t2.num_elements = 6;
            w.add_tensor(t2);
            TensorExportInfo t3;
            t3.name = "blk.0.q";
            t3.shape = {1, 32};
            t3.type = GGMLType::Q8_0;
            t3.data = q;
            t3.num_elements = 32;
            w.add_tensor(t3);
            EXPECT(w.tensor_count() == 3);

            // Refusals: shape and count disagree; ne0 not whole blocks; no data.
            TensorExportInfo bad = t2;
            bad.num_elements = 5;
            EXPECT(throws([&] { w.add_tensor(bad); }));
            bad = t3;
            bad.shape = {2, 16};
            EXPECT(throws([&] { w.add_tensor(bad); }));
            bad = t1;
            bad.data = nullptr;
            EXPECT(throws([&] { w.add_tensor(bad); }));
            EXPECT(w.tensor_count() == 3);

            w.set_pad_final_tensor(pad_end);
            EXPECT(w.finalize());
        }

        Bytes e;
        e.raw<uint32_t>(0x46554747u);
        e.raw<uint32_t>(3);
        e.raw<uint64_t>(3);
        e.raw<uint64_t>(16);
        e.key("general.architecture", 8);
        e.str("llama");
        e.key("general.alignment", 4);
        e.raw<uint32_t>(32);
        e.key("t.u8", 0);
        e.raw<uint8_t>(200);
        e.key("t.i8", 1);
        e.raw<int8_t>(-3);
        e.key("t.u16", 2);
        e.raw<uint16_t>(60000);
        e.key("t.i16", 3);
        e.raw<int16_t>(-30000);
        e.key("t.i32", 5);
        e.raw<int32_t>(-7);
        e.key("t.u64", 10);
        e.raw<uint64_t>(1ull << 40);
        e.key("t.i64", 11);
        e.raw<int64_t>(-(1ll << 40));
        e.key("t.f32", 6);
        e.raw<float>(0.25f);
        e.key("t.f64", 12);
        e.raw<double>(-0.125);
        e.key("t.bool", 7);
        e.raw<uint8_t>(1);
        e.key("t.strs", 9);
        e.raw<uint32_t>(8);
        e.raw<uint64_t>(3);
        e.str("a");
        e.str("");
        e.str("xyz");
        e.key("t.f32s", 9);
        e.raw<uint32_t>(6);
        e.raw<uint64_t>(2);
        e.raw<float>(1.0f);
        e.raw<float>(-2.0f);
        e.key("t.i32s", 9);
        e.raw<uint32_t>(5);
        e.raw<uint64_t>(3);
        e.raw<int32_t>(3);
        e.raw<int32_t>(-4);
        e.raw<int32_t>(5);
        e.key("t.u32s", 9);
        e.raw<uint32_t>(4);
        e.raw<uint64_t>(0);
        // tensor infos: dims fastest first, offsets from the data start
        e.str("vec");
        e.raw<uint32_t>(1);
        e.raw<uint64_t>(5);
        e.raw<uint32_t>(0);
        e.raw<uint64_t>(0);
        e.str("mat");
        e.raw<uint32_t>(2);
        e.raw<uint64_t>(2);
        e.raw<uint64_t>(3);
        e.raw<uint32_t>(0);
        e.raw<uint64_t>(32);
        e.str("blk.0.q");
        e.raw<uint32_t>(2);
        e.raw<uint64_t>(32);
        e.raw<uint64_t>(1);
        e.raw<uint32_t>(8);
        e.raw<uint64_t>(64);
        e.zeros_to(32);
        const size_t data_start = e.b.size();
        e.b.insert(e.b.end(), reinterpret_cast<const uint8_t*>(a), reinterpret_cast<const uint8_t*>(a) + 20);
        e.zeros_to(32);
        e.b.insert(e.b.end(), reinterpret_cast<const uint8_t*>(m), reinterpret_cast<const uint8_t*>(m) + 24);
        e.zeros_to(32);
        e.b.insert(e.b.end(), q, q + 34);
        if (pad_end) e.zeros_to(32);

        const std::vector<uint8_t> got = slurp(path);
        std::remove(path.c_str());
        EXPECT(data_start % 32 == 0);
        EXPECT(got.size() == e.b.size());
        EXPECT(got == e.b);
        std::printf("  %s: %zu bytes, data at %zu, %s the hand-assembled file\n",
                    pad_end ? "final tensor padded" : "plain              ", got.size(), data_start,
                    got == e.b ? "identical to" : "DIFFERENT from");
    }

    // An empty file: header only, padded to the alignment.
    {
        const std::string path = tmp + "/gguf_writer_test_empty.gguf";
        GGUFWriter w(path);
        EXPECT(w.finalize());
        const std::vector<uint8_t> got = slurp(path);
        std::remove(path.c_str());
        EXPECT(got.size() == 32);
    }

    // An unwritable path fails cleanly.
    {
        GGUFWriter w(tmp + "/no_such_directory_c3/x.gguf");
        EXPECT(!w.finalize());
    }

    if (failures) {
        std::printf("\n[FAIL] %d checks failed\n", failures);
        return 1;
    }
    std::printf("\n[PASS] gguf_writer.hpp\n");
    return 0;
}
