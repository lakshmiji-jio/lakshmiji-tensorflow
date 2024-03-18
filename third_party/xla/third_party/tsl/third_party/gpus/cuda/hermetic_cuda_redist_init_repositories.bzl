# Copyright 2024 The TensorFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hermetic CUDA repositories initialization. Consult the WORKSPACE on how to use it."""

load("//third_party:repo.bzl", "tf_mirror_urls")
load("//third_party/gpus:hermetic_cuda_configure.bzl", "hermetic_cuda_configure")
load("//third_party/nccl:hermetic_nccl_configure.bzl", "hermetic_nccl_configure")

_OS_ARCH_DICT = {
    "amd64": "x86_64-unknown-linux-gnu",
    "aarch64": "aarch64-unknown-linux-gnu",
}
_REDIST_ARCH_DICT = {
    "linux-x86_64": "x86_64-unknown-linux-gnu",
    "linux-sbsa": "aarch64-unknown-linux-gnu",
}

CUDA_DIST_PATH_PREFIX = "https://developer.download.nvidia.com/compute/cuda/redist/"
CUDNN_DIST_PATH_PREFIX = "https://developer.download.nvidia.com/compute/cudnn/redist/"

# The versions are different for x86 and aarch64 architectures because only
# NCCL release versions 2.20.3 and 2.20.5 have the wheels for aarch64.
_CUDA_12_NCCL_WHEEL_DICT = {
    "x86_64-unknown-linux-gnu": {
        "url": "https://files.pythonhosted.org/packages/38/00/d0d4e48aef772ad5aebcf70b73028f88db6e5640b36c38e90445b7a57c45/nvidia_nccl_cu12-2.19.3-py3-none-manylinux1_x86_64.whl",
        "sha256": "a9734707a2c96443331c1e48c717024aa6678a0e2a4cb66b2c364d18cee6b48d",
    },
    "aarch64-unknown-linux-gnu": {
        "url": "https://files.pythonhosted.org/packages/a8/e9/7f599dd20bedf3f53f4ae71f7f66db3a2c4164e98355a8dae2177c41ba1d/nvidia_nccl_cu12-2.20.3-py3-none-manylinux2014_aarch64.whl",
        "sha256": "1fd5390c3820eacb8dca6bb4a7ce3c84bca0cbe43d20a34c7951f0230c2061c7",
    },
}

_CUDA_11_NCCL_WHEEL_DICT = {
    "x86_64-unknown-linux-gnu": {
        "url": "https://files.pythonhosted.org/packages/0e/7d/cc3dbf36c5af39b042d508b7a441ada1fce69bd18c800e5c25dc4e9f8933/nvidia_nccl_cu11-2.19.3-py3-none-manylinux1_x86_64.whl",
        "sha256": "7c58afbeddf7f7c6b7dd7d84a7f4e85462610ee0c656287388b96d89dcf046d5",
    },
}

CUDA_NCCL_WHEELS = {
    "11.8": _CUDA_11_NCCL_WHEEL_DICT,
    "12.1.1": _CUDA_12_NCCL_WHEEL_DICT,
    "12.3.1": _CUDA_12_NCCL_WHEEL_DICT,
    "12.3.2": _CUDA_12_NCCL_WHEEL_DICT,
}

_SUPPORTED_ARCHIVE_EXTENSIONS = [".zip", ".jar", ".war", ".aar", ".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.zst", ".tzst", ".tar.bz2", ".tbz", ".ar", ".deb", ".whl"]

def _get_env_var(ctx, name):
    if name in ctx.os.environ:
        return ctx.os.environ[name]
    else:
        return None

def _get_file_name(url):
    last_slash_index = url.rfind("/")
    return url[last_slash_index + 1:]

def _get_archive_name(url):
    filename = _get_file_name(url)
    for extension in _SUPPORTED_ARCHIVE_EXTENSIONS:
        if filename.endswith(extension):
            return filename[:-len(extension)]
    return filename

def _cuda_http_archive_impl(repository_ctx):
    cuda_or_cudnn_version = None
    dist_version = ""
    saved_version = ""
    cuda_version = (_get_env_var(repository_ctx, "HERMETIC_CUDA_VERSION") or
                    _get_env_var(repository_ctx, "TF_CUDA_VERSION"))
    cudnn_version = (_get_env_var(repository_ctx, "HERMETIC_CUDNN_VERSION") or
                     _get_env_var(repository_ctx, "TF_CUDNN_VERSION"))
    if repository_ctx.attr.is_cudnn_dist:
        cuda_or_cudnn_version = cudnn_version
    else:
        cuda_or_cudnn_version = cuda_version
    if cuda_or_cudnn_version:
        # Download archive only when GPU config is used.
        dist_version = repository_ctx.attr.dist_version
        arch_key = _OS_ARCH_DICT[repository_ctx.os.arch]
        if arch_key not in repository_ctx.attr.url_dict.keys():
            arch_key = "cuda{version}_{arch}".format(
                version = cuda_version.split(".")[0],
                arch = arch_key,
            )
        if arch_key in repository_ctx.attr.url_dict.keys():
            (url, sha256) = repository_ctx.attr.url_dict[arch_key]

            # If url is not relative, then appending prefix is not needed.
            if not (url.startswith("http") or url.startswith("file:///")):
                if repository_ctx.attr.is_cudnn_dist:
                    url = repository_ctx.attr.cudnn_dist_path_prefix + url
                else:
                    url = repository_ctx.attr.cuda_dist_path_prefix + url
            archive_name = _get_archive_name(url)
            file_name = _get_file_name(url)

            repository_ctx.download(
                url = tf_mirror_urls(url),
                output = file_name,
                sha256 = sha256,
            )
            if repository_ctx.attr.override_strip_prefix:
                strip_prefix = repository_ctx.attr.override_strip_prefix
            else:
                strip_prefix = archive_name
            repository_ctx.extract(
                archive = file_name,
                stripPrefix = strip_prefix,
            )
            repository_ctx.delete(file_name)

            if repository_ctx.attr.build_template:
                saved_version = dist_version.split(".")[0] if dist_version else ""
                build_template = repository_ctx.attr.build_template

                # Workaround for CUDA 11 distribution versions.
                if cuda_version and cuda_version.startswith("11"):
                    if saved_version == "11":
                        if repository_ctx.name == "cuda_cudart":
                            saved_version = "11.0"
                        if repository_ctx.name == "cuda_cupti":
                            saved_version = cuda_version

                repository_ctx.template(
                    "BUILD",
                    build_template,
                    {"%{version}": saved_version},
                )
            else:
                repository_ctx.file(
                    "BUILD",
                    repository_ctx.read(repository_ctx.attr.build_file),
                )
        else:
            # If no matching arch is found, use the dummy build file.
            repository_ctx.file(
                "BUILD",
                repository_ctx.read(repository_ctx.attr.dummy_build_file),
            )
    else:
        # If no CUDA or CUDNN version is found, use the dummy build file if present.
        repository_ctx.file(
            "BUILD",
            repository_ctx.read(repository_ctx.attr.dummy_build_file or
                                repository_ctx.attr.build_file),
        )
    repository_ctx.file("version.txt", saved_version)

_cuda_http_archive = repository_rule(
    implementation = _cuda_http_archive_impl,
    attrs = {
        "dist_version": attr.string(mandatory = True),
        "url_dict": attr.string_list_dict(mandatory = True),
        "build_template": attr.label(),
        "dummy_build_file": attr.label(),
        "build_file": attr.label(),
        "is_cudnn_dist": attr.bool(),
        "override_strip_prefix": attr.string(),
        "cudnn_dist_path_prefix": attr.string(),
        "cuda_dist_path_prefix": attr.string(),
    },
    environ = [
        "HERMETIC_CUDA_VERSION",
        "HERMETIC_CUDNN_VERSION",
        "TF_CUDA_VERSION",
        "TF_CUDNN_VERSION",
    ],
)

def cuda_http_archive(name, dist_version, url_dict, **kwargs):
    _cuda_http_archive(
        name = name,
        dist_version = dist_version,
        url_dict = url_dict,
        **kwargs
    )

def _cuda_wheel_impl(repository_ctx):
    cuda_version = (_get_env_var(repository_ctx, "HERMETIC_CUDA_VERSION") or
                    _get_env_var(repository_ctx, "TF_CUDA_VERSION"))
    if cuda_version:
        # Download archive only when GPU config is used.
        arch = _OS_ARCH_DICT[repository_ctx.os.arch]
        dict_key = "{cuda_version}-{arch}".format(
            cuda_version = cuda_version,
            arch = arch,
        )
        supported_versions = repository_ctx.attr.url_dict.keys()
        if dict_key not in supported_versions:
            fail(
                ("The supported NCCL versions are {supported_versions}." +
                 " Please provide a supported version in HERMETIC_CUDA_VERSION" +
                 " environment variable or add NCCL distribution for" +
                 " CUDA version={version}, OS={arch}.")
                    .format(
                    supported_versions = supported_versions,
                    version = cuda_version,
                    arch = arch,
                ),
            )
        sha256 = repository_ctx.attr.sha256_dict[dict_key]
        url = repository_ctx.attr.url_dict[dict_key]

        archive_name = _get_archive_name(url)
        file_name = archive_name + ".zip"

        repository_ctx.download(
            url = tf_mirror_urls(url),
            output = file_name,
            sha256 = sha256,
        )
        repository_ctx.extract(
            archive = file_name,
            stripPrefix = repository_ctx.attr.strip_prefix,
        )
        repository_ctx.delete(file_name)

        repository_ctx.file(
            "BUILD",
            repository_ctx.read(repository_ctx.attr.build_file),
        )
    else:
        # If no CUDA version is found, use the dummy build file if present.
        repository_ctx.file(
            "BUILD",
            repository_ctx.read(repository_ctx.attr.dummy_build_file or
                                repository_ctx.attr.build_file),
        )

_cuda_wheel = repository_rule(
    implementation = _cuda_wheel_impl,
    attrs = {
        "sha256_dict": attr.string_dict(mandatory = True),
        "url_dict": attr.string_dict(mandatory = True),
        "build_file": attr.label(),
        "dummy_build_file": attr.label(),
        "strip_prefix": attr.string(),
    },
    environ = ["HERMETIC_CUDA_VERSION", "TF_CUDA_VERSION"],
)

def cuda_wheel(name, sha256_dict, url_dict, **kwargs):
    _cuda_wheel(
        name = name,
        sha256_dict = sha256_dict,
        url_dict = url_dict,
        **kwargs
    )

def _get_distribution_urls(dist_info):
    url_dict = {}
    for arch in _REDIST_ARCH_DICT.keys():
        if "relative_path" not in dist_info[arch]:
            if "full_path" not in dist_info[arch]:
                for cuda_version, data in dist_info[arch].items():
                    # CUDNN JSON might contain paths for each CUDA version.
                    path_key = "relative_path"
                    if path_key not in data.keys():
                        path_key = "full_path"
                    url_dict["{cuda_version}_{arch}" \
                        .format(
                        cuda_version = cuda_version,
                        arch = _REDIST_ARCH_DICT[arch],
                    )] = [data[path_key], data.get("sha256", "")]
            else:
                url_dict[_REDIST_ARCH_DICT[arch]] = [
                    dist_info[arch]["full_path"],
                    dist_info[arch].get("sha256", ""),
                ]
        else:
            url_dict[_REDIST_ARCH_DICT[arch]] = [
                dist_info[arch]["relative_path"],
                dist_info[arch].get("sha256", ""),
            ]
    return url_dict

def _get_cuda_archive(
        repo_name,
        dist_dict,
        dist_name,
        cuda_dist_path_prefix = CUDA_DIST_PATH_PREFIX,
        cudnn_dist_path_prefix = CUDNN_DIST_PATH_PREFIX,
        build_file = None,
        build_template = None,
        dummy_build_file = None,
        is_cudnn_dist = False):
    if dist_name in dist_dict.keys():
        return cuda_http_archive(
            name = repo_name,
            dist_version = dist_dict[dist_name]["version"],
            build_file = build_file,
            build_template = build_template,
            dummy_build_file = dummy_build_file,
            url_dict = _get_distribution_urls(dist_dict[dist_name]),
            is_cudnn_dist = is_cudnn_dist,
            cuda_dist_path_prefix = cuda_dist_path_prefix,
            cudnn_dist_path_prefix = cudnn_dist_path_prefix,
        )
    else:
        return cuda_http_archive(
            name = repo_name,
            dist_version = "",
            build_file = build_file,
            build_template = build_template,
            dummy_build_file = dummy_build_file,
            url_dict = {"": []},
            is_cudnn_dist = is_cudnn_dist,
        )

def _all_cuda_distributions(
        cuda_nccl_wheels,
        cuda_distributions,
        cudnn_distributions,
        cuda_dist_path_prefix,
        cudnn_dist_path_prefix):
    nccl_artifacts_dict = {"sha256_dict": {}, "url_dict": {}}
    for cuda_version, nccl_wheel_info in cuda_nccl_wheels.items():
        for arch in _OS_ARCH_DICT.values():
            if arch in nccl_wheel_info.keys():
                cuda_version_to_arch_key = "%s-%s" % (cuda_version, arch)
                nccl_artifacts_dict["sha256_dict"][cuda_version_to_arch_key] = nccl_wheel_info[arch].get("sha256", "")
                nccl_artifacts_dict["url_dict"][cuda_version_to_arch_key] = nccl_wheel_info[arch]["url"]

    cuda_wheel(
        name = "cuda_nccl",
        sha256_dict = nccl_artifacts_dict["sha256_dict"],
        url_dict = nccl_artifacts_dict["url_dict"],
        build_file = Label("//third_party/gpus/cuda:cuda_nccl.BUILD"),
        dummy_build_file = Label("//third_party/gpus/cuda:cuda_nccl_dummy.BUILD"),
        strip_prefix = "nvidia/nccl",
    )

    cudnn_build_template = Label("//third_party/gpus/cuda:cuda_cudnn.BUILD.tpl")
    cudnn_dummy_build_file = Label("//third_party/gpus/cuda:cuda_cudnn_dummy.BUILD")
    if "cudnn" in cudnn_distributions.keys():
        cudnn_version = cudnn_distributions["cudnn"]["version"]
        if cudnn_version.startswith("9"):
            cudnn_build_template = Label("//third_party/gpus/cuda:cuda_cudnn9.BUILD.tpl")
            cudnn_dummy_build_file = Label("//third_party/gpus/cuda:cuda_cudnn9_dummy.BUILD")
    _get_cuda_archive(
        repo_name = "cuda_cudnn",
        dist_dict = cudnn_distributions,
        dist_name = "cudnn",
        build_template = cudnn_build_template,
        dummy_build_file = cudnn_dummy_build_file,
        is_cudnn_dist = True,
        cudnn_dist_path_prefix = cudnn_dist_path_prefix,
    )

    _get_cuda_archive(
        repo_name = "cuda_cccl",
        dist_dict = cuda_distributions,
        dist_name = "cuda_cccl",
        build_file = Label("//third_party/gpus/cuda:cuda_cccl.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_cublas",
        dist_dict = cuda_distributions,
        dist_name = "libcublas",
        build_template = Label("//third_party/gpus/cuda:cuda_cublas.BUILD.tpl"),
        dummy_build_file = Label("//third_party/gpus/cuda:cuda_cublas_dummy.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_cudart",
        dist_dict = cuda_distributions,
        dist_name = "cuda_cudart",
        build_template = Label("//third_party/gpus/cuda:cuda_cudart.BUILD.tpl"),
        dummy_build_file = Label("//third_party/gpus/cuda:cuda_cudart_dummy.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_cufft",
        dist_dict = cuda_distributions,
        dist_name = "libcufft",
        build_template = Label("//third_party/gpus/cuda:cuda_cufft.BUILD.tpl"),
        dummy_build_file = Label("//third_party/gpus/cuda:cuda_cufft_dummy.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_cupti",
        dist_dict = cuda_distributions,
        dist_name = "cuda_cupti",
        build_template = Label("//third_party/gpus/cuda:cuda_cupti.BUILD.tpl"),
        dummy_build_file = Label("//third_party/gpus/cuda:cuda_cupti_dummy.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_curand",
        dist_dict = cuda_distributions,
        dist_name = "libcurand",
        build_template = Label("//third_party/gpus/cuda:cuda_curand.BUILD.tpl"),
        dummy_build_file = Label("//third_party/gpus/cuda:cuda_curand_dummy.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_cusolver",
        dist_dict = cuda_distributions,
        dist_name = "libcusolver",
        build_template = Label("//third_party/gpus/cuda:cuda_cusolver.BUILD.tpl"),
        dummy_build_file = Label("//third_party/gpus/cuda:cuda_cusolver_dummy.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_cusparse",
        dist_dict = cuda_distributions,
        dist_name = "libcusparse",
        build_template = Label("//third_party/gpus/cuda:cuda_cusparse.BUILD.tpl"),
        dummy_build_file = Label("//third_party/gpus/cuda:cuda_cusparse_dummy.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_nvcc",
        dist_dict = cuda_distributions,
        dist_name = "cuda_nvcc",
        build_file = Label("//third_party/gpus/cuda:cuda_nvcc.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_nvjitlink",
        dist_dict = cuda_distributions,
        dist_name = "libnvjitlink",
        build_template = Label("//third_party/gpus/cuda:cuda_nvjitlink.BUILD.tpl"),
        dummy_build_file = Label("//third_party/gpus/cuda:cuda_nvjitlink_dummy.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_nvml",
        dist_dict = cuda_distributions,
        dist_name = "cuda_nvml_dev",
        build_file = Label("//third_party/gpus/cuda:cuda_nvml.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_nvprune",
        dist_dict = cuda_distributions,
        dist_name = "cuda_nvprune",
        build_file = Label("//third_party/gpus/cuda:cuda_nvprune.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_nvrtc",
        dist_dict = cuda_distributions,
        dist_name = "cuda_nvrtc",
        build_template = Label("//third_party/gpus/cuda:cuda_nvrtc.BUILD.tpl"),
        dummy_build_file = Label("//third_party/gpus/cuda:cuda_nvrtc_dummy.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )
    _get_cuda_archive(
        repo_name = "cuda_nvtx",
        dist_dict = cuda_distributions,
        dist_name = "cuda_nvtx",
        build_file = Label("//third_party/gpus/cuda:cuda_nvtx.BUILD"),
        cuda_dist_path_prefix = cuda_dist_path_prefix,
    )

def hermetic_cuda_redist_init_repositories(
        cuda_nccl_wheels,
        cuda_distributions,
        cudnn_distributions,
        cuda_dist_path_prefix,
        cudnn_dist_path_prefix):
    _all_cuda_distributions(
        cuda_nccl_wheels = cuda_nccl_wheels,
        cuda_distributions = cuda_distributions,
        cudnn_distributions = cudnn_distributions,
        cuda_dist_path_prefix = cuda_dist_path_prefix,
        cudnn_dist_path_prefix = cudnn_dist_path_prefix,
    )

    hermetic_cuda_configure(name = "local_config_cuda")
    hermetic_nccl_configure(name = "local_config_nccl")
