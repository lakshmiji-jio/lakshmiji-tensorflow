licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])

cc_library(
    name = "cudnn",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudnn_ops",
)

cc_library(
    name = "cudnn_cnn",
)

cc_library(
    name = "cudnn_adv",
)

cc_library(
    name = "cudnn_graph",
)

cc_library(
    name = "cudnn_engines_precompiled",
)

cc_library(
    name = "cudnn_engines_runtime_compiled",
)

cc_library(
    name = "cudnn_heuristic",
)

cc_library(
    name = "cudnn_other",
    visibility = ["//visibility:public"],
    deps = [
        ":cudnn_adv",
        ":cudnn_cnn",
        ":cudnn_engines_precompiled",
        ":cudnn_engines_runtime_compiled",
        ":cudnn_graph",
        ":cudnn_heuristic",
        ":cudnn_ops",
    ],
)

cc_library(
    name = "headers",
    hdrs = glob([
        "include/**",
    ]),
    include_prefix = "third_party/gpus/cudnn",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
