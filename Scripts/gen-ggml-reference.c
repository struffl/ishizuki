#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ggml.h"

static uint64_t s = 0x9E3779B97F4A7C15ULL;
static uint8_t rnd(void) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return (uint8_t)(s >> 24);
}

static const enum ggml_type types[] = {
    GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
    GGML_TYPE_Q2_K, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K,
    GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
    GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS,
};

int main(int argc, char **argv) {
    struct ggml_init_params ip = { 1024*1024, NULL, false };
    struct ggml_context *ctx = ggml_init(ip);
    (void)ctx;
    const int nblocks = 8;
    FILE *f = fopen(argv[1], "wb");
    uint32_t n = sizeof(types)/sizeof(types[0]);
    fwrite("GGMLREF1", 1, 8, f);
    fwrite(&n, 4, 1, f);

    for (uint32_t i = 0; i < n; i++) {
        enum ggml_type t = types[i];
        const struct ggml_type_traits *tr = ggml_get_type_traits(t);
        int bs = ggml_blck_size(t);
        size_t ts = ggml_type_size(t);
        int elems = nblocks * bs;
        size_t raw = nblocks * ts;

        uint8_t *bytes = malloc(raw);
        for (size_t j = 0; j < raw; j++) bytes[j] = rnd();
        if (t == GGML_TYPE_F32) {
            float *fp = (float *)bytes;
            for (int j = 0; j < elems; j++) fp[j] = (float)((int)rnd() - 128) * 0.03125f;
        }
        float *out = malloc(elems * sizeof(float));
        if (tr->to_float) tr->to_float(bytes, out, elems);
        else memcpy(out, bytes, elems * sizeof(float));

        uint32_t tid = (uint32_t)t, nb = nblocks, rb = (uint32_t)raw, nf = elems;
        fwrite(&tid, 4, 1, f); fwrite(&nb, 4, 1, f);
        fwrite(&rb, 4, 1, f); fwrite(&nf, 4, 1, f);
        fwrite(bytes, 1, raw, f);
        fwrite(out, 4, elems, f);
        printf("%-8s blocks=%d raw=%zu elems=%d  first=%.6f %.6f\n",
               ggml_type_name(t), nblocks, raw, elems, out[0], out[1]);
        free(bytes); free(out);
    }
    fclose(f);
    return 0;
}
