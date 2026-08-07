PORTNAME=	ggml
DISTVERSIONPREFIX=	v
DISTVERSION=	0.19.0
CATEGORIES=	misc # machine-learning

MAINTAINER=	yuri@FreeBSD.org
COMMENT=	Tensor library for machine learning
WWW=		https://github.com/ggml-org/ggml

LICENSE=	MIT
LICENSE_FILE=	${WRKSRC}/LICENSE

BROKEN_i386=	compilation fails: LLVM ERROR: out of memory

USES=		cmake:testing compiler:c++17-lang llvm:build,min=22 pkgconfig python:run shebangfix
USE_LDCONFIG=	yes

USE_GITHUB=	yes
GH_ACCOUNT=	ggml-org

CMAKE_ON=	BUILD_SHARED_LIBS \
		GGML_CPU \
		GGML_SSE42 \
		GGML_AVX \
		GGML_AVX-VNNI \
		GGML_AVX2 \
		GGML_BMI2 \
		GGML_FMA \
		GGML_F16C

CMAKE_OFF=	GGML_NATIVE \
		GGML_BUILD_TESTS \
		GGML_BUILD_EXAMPLES \
		GGML_AVX512F \
		GGML_AVX512_VBMI \
		GGML_AVX512_VNNI \
		GGML_AVX512_BF16 \
		GGML_AMX_TILE \
		GGML_AMX_INT8 \
		GGML_AMX_BF16 \
		GGML_OPENCL_USE_ADRENO_KERNELS

CMAKE_TESTING_ON=	GGML_BUILD_TESTS

BINARY_ALIAS=	git=false

PLIST_SUB+=	DISTVERSION=${DISTVERSION}

OPTIONS_RADIO=		BLAS
OPTIONS_RADIO_BLAS=	OPENBLAS BLIS
OPTIONS_DEFINE=		OPENCL VULKAN
OPTIONS_DEFAULT=	OPENBLAS OPENCL VULKAN
OPTIONS_SUB=		yes

VULKAN_CMAKE_BOOL=	GGML_VULKAN
VULKAN_BUILD_DEPENDS=	glslc:graphics/shaderc \
			spirv-headers>0:graphics/spirv-headers \
			vulkan-headers>0:graphics/vulkan-headers
VULKAN_LIB_DEPENDS=	libvulkan.so:graphics/vulkan-loader

OPENCL_DESC=		Add OpenCL Support
OPENCL_CMAKE_BOOL=	GGML_OPENCL -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF
OPENCL_BUILD_DEPENDS=	opencl>=0:devel/opencl \
			ocl-icd>0:devel/ocl-icd
OPENCL_RUN_DEPENDS=	opencl>=0:devel/opencl \
			ocl-icd>0:devel/ocl-icd

OPENBLAS_LIB_DEPENDS=	libopenblas.so:math/openblas
OPENBLAS_CMAKE_ON=	-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS

BLIS_DESC=		BLAS-like libraries from BLIS
BLIS_LIB_DEPENDS=	libblis.so:math/blis
BLIS_CMAKE_ON=		-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=FLAME

.include <bsd.port.options.mk>

.if ${PORT_OPTIONS:MOPENBLAS} || ${PORT_OPTIONS:MBLIS}
PLIST_SUB+=		BLAS=""
SUB_LIST+=		BLAS=""
.else
PLIST_SUB+=		BLAS="@comment "
SUB_LIST+=		BLAS="@comment "
.endif

.if ${PORT_OPTIONS:MOPENCL}
PLIST_SUB+=		OPENCL=""
SUB_LIST+=		OPENCL=""
.else
PLIST_SUB+=		OPENCL="@comment "
SUB_LIST+=		OPENCL="@comment "
.endif

# tests as of 0.13.1: 100% tests passed, 0 tests failed out of 21

post-patch:
	@${REINPLACE_CMD} \
		-E 's,^(#pragma OPENCL EXTENSION cl_khr_subgroups),//\1,g' ${WRKSRC}/src/ggml-opencl/kernels/*.cl
	@${REINPLACE_CMD} \
		-e 's,#ifdef INTEL_GPU,#if !defined(ADRENO_GPU),g' ${WRKSRC}/src/ggml-opencl/kernels/*.cl

.include <bsd.port.mk>
