licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])

cc_library(
    name = "cudnn",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudnn_ops_infer",
)

cc_library(
    name = "cudnn_cnn_infer",
)

cc_library(
    name = "cudnn_ops_train",
)

cc_library(
    name = "cudnn_cnn_train",
)

cc_library(
    name = "cudnn_adv_infer",
)

cc_library(
    name = "cudnn_adv_train",
)

cc_library(
    name = "cudnn_other",
    visibility = ["//visibility:public"],
    deps = [
        ":cudnn_adv_infer",
        ":cudnn_adv_train",
        ":cudnn_cnn_infer",
        ":cudnn_cnn_train",
        ":cudnn_ops_infer",
        ":cudnn_ops_train",
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
