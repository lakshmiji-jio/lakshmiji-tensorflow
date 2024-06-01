licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])

cc_import(
    name = "cudnn",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn.so.%{version}",
    visibility = ["//visibility:public"],
)

cc_import( 
    name = "cudnn_ops",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_ops.so.%{version}",
)

cc_import( 
    name = "cudnn_cnn",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_cnn.so.%{version}",
)

cc_import( 
    name = "cudnn_adv",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_adv.so.%{version}",
)

cc_import( 
    name = "cudnn_graph",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_graph.so.%{version}",
)

cc_import(
    name = "cudnn_engines_precompiled",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_engines_precompiled.so.%{version}",
)

cc_import(
    name = "cudnn_engines_runtime_compiled",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_engines_runtime_compiled.so.%{version}",
)

cc_import(
    name = "cudnn_heuristic",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_heuristic.so.%{version}",
)

cc_library(
    name = "cudnn_other",
    deps = [
      ":cudnn_ops", 
      ":cudnn_cnn", 
      ":cudnn_adv", 
      ":cudnn_graph", 
      ":cudnn_engines_precompiled", 
      ":cudnn_engines_runtime_compiled", 
      ":cudnn_heuristic"
    ],
    visibility = ["//visibility:public"],
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
