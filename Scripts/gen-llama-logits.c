// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT
//
// llama.cpp's own logits for one prompt, so this runtime can be checked against the reference
// implementation reading the very same file.
//
//   clang -O2 -o genlogits Scripts/gen-llama-logits.c \
//     -I <llama.cpp>/include -I <llama.cpp>/ggml/include \
//     -L <llama.cpp>/build/bin -lllama -Wl,-rpath,<llama.cpp>/build/bin
//   ./genlogits model.gguf "the prompt" out.bin

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "llama.h"

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.gguf prompt out.bin\n", argv[0]); return 1; }

    llama_backend_init();

    struct llama_model_params mp = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_token tokens[512];
    const int n = llama_tokenize(vocab, argv[2], (int32_t)strlen(argv[2]), tokens, 512, false, false);
    if (n < 1) { fprintf(stderr, "tokenize failed (%d)\n", n); return 1; }

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024;
    cp.n_batch = 512;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to make a context\n"); return 1; }

    if (llama_decode(ctx, llama_batch_get_one(tokens, n)) != 0) {
        fprintf(stderr, "decode failed\n"); return 1;
    }
    const float *logits = llama_get_logits_ith(ctx, n - 1);

    // "LLAMALG1", token count, vocab size, the tokens, then the last position's logits.
    FILE *f = fopen(argv[3], "wb");
    fwrite("LLAMALG1", 1, 8, f);
    uint32_t a = (uint32_t)n, b = (uint32_t)n_vocab;
    fwrite(&a, 4, 1, f);
    fwrite(&b, 4, 1, f);
    for (int i = 0; i < n; i++) { int32_t t = tokens[i]; fwrite(&t, 4, 1, f); }
    fwrite(logits, sizeof(float), n_vocab, f);
    fclose(f);

    fprintf(stderr, "wrote %d tokens and %d logits to %s\n", n, n_vocab, argv[3]);
    return 0;
}
