licenses(["restricted"])  # NVIDIA proprietary license

cc_import(
    name = "nvrtc",
    hdrs = [":headers"],
    shared_library = "lib/libnvrtc.so.%{version}",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    hdrs = glob([
        "include/**",
    ]),
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
